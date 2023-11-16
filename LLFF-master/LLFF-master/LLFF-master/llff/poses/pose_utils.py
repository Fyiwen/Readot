import numpy as np
import os
import sys
import imageio
import skimage.transform

from llff.poses.colmap_wrapper import run_colmap
import llff.poses.colmap_read_model as read_model


def load_colmap_data(realdir): # 输入colmap处理的内容所在的路径（项目路径）
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin') # cameras.bin所在路径
    camdata = read_model.read_cameras_binary(camerasfile) # 以一坨一坨字典的形式，得到所有的相机参数信息。有28大坨对应28张图的28个相机信息
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]] # 得到关于第一个相机的完整参数信息
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0] #从相机参数中得到hwf信息
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin') #images.bin所在路径
    imdata = read_model.read_images_binary(imagesfile)  # 得到28坨图像参数的信息，每坨里都是字典形式。坨也是字典形式
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata] # 得到每个图片的名称（28个，['1.jpg', '2.jpg', '3.jpg', '31.jpg', '34.jpg', '35.jpg', '36.jpg', '37.jpg', '38.jpg', '39.jpg', '4.jpg', '40.jpg', '41.jpg', '42.jpg', '43.jpg', '44.jpg', '45.jpg', '46.jpg', '47.jpg', '48.jpg', '49.jpg', '5.jpg', '50.jpg', '51.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']）
    for i in np.argsort(names): # 这两行是另外加的，会输出那些图片和位姿匹配上了的（在colmap处理中匹配上的）图片名字
        print(names[i], end=' ')
    print( 'Images #', len(names))
    perm = np.argsort(names) # 按照数组 names 中元素的值进行排序，并返回排序后的索引数组
    for k in imdata: # 遍历每一个图片对应的参数们
        im = imdata[k] # 这里存这一轮的所有参数信息
        R = im.qvec2rotmat()  # 提取得到这个图片的旋转信息（3，3）
        t = im.tvec.reshape([3,1])  # 提取得到这个图片的平移信息（3，1）
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0) # 旋转+平移链接作为变换信息，再变成齐次的表示。（4，4）
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0) # (28,4,4)所有w2c矩阵
    c2w_mats = np.linalg.inv(w2c_mats) # 求逆算出对应c2w矩阵（28，4，4）
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0]) # （3，4，28）主要是3*4的姿态矩阵
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1) # （3，5，28）主要是加上hwf后的3*4矩阵
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin') # points3D文件的路径
    pts3d = read_model.read_points3d_binary(points3dfile) # # 以一坨一坨字典的形式，得到所有的3D点信息
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm  # 返回所有相机姿态矩阵（3，5，28），3D点信息（4404）和图片名字排序后的索引（28）


def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:  # 遍历每一个3D点
        pts_arr.append(pts3d[k].xyz) # 提取当前点的xyz信息存到这个变量中
        cams = [0] * poses.shape[-1] # 一个全是0的列表，0的总个数为相机姿态的总个数[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for ind in pts3d[k].image_ids: # 遍历这个点对应的图像的索引
            if len(cams) < ind - 1: #因为这个图片索引值是基于一共有28张图片匹配上了之后得到的，所以索引值一定小于28.那么一旦不满足就代表出现了没有匹配成功的图片就要报错
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1 # 在对应索引处标记为1，表示这个索引处对应的图片能匹配上这个点
        vis_arr.append(cams) # 得到这个变量，结果是大列表嵌小列表，小列表28位，例如第一个小列表中为1的位置索引对应点1对应的那几张图片的索引位置

    pts_arr = np.array(pts_arr) # (4404,3)所有点和他对应的xyz坐标
    vis_arr = np.array(vis_arr) # (4404,28)所有点和他能匹配上的图片的标记
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    # 先计算所有点，以一个点为例，因为一个点对应28个姿态矩阵（虽然有些不是对应的），拿这个点的xyz坐标减去姿态矩阵平移量，结果乘上姿态矩阵z轴分量，得到深度值，一个点28个深度值，但是这28个深度值有些是不对的，就用第二行来做筛选。只取这个点对应的图片的姿态矩阵求出来的深度值，才是有用的深度值
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0) # （4404，28）求出所有3D点的深度值
    valid_z = zvals[vis_arr==1] #（16430）把有效的深度值取出来。何为有效？点对应的图片的姿态矩阵求出来的才是
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    for i in perm: # 遍历每一个索引
        vis = vis_arr[:, i] # （4404）得到所有点的匹配标记中，索引位置为i的标记值
        zs = zvals[:, i] #（4404）取所有点的关于索引i的对应深度值
        zs = zs[vis==1] # （856）取出有效的深度值。也就是说这个索引i表示的图片，既与某点匹配，而且这个深度值也是这样的一个匹配情况对应的。
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9) # 从zs中计算一个数使得zs中0.1%（有0.1*856个）数小于等于他，再算一个数使得zs里面99。9%的数小于等于他
        # print( i, close_depth, inf_depth )
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0)) # 把这个索引对应的图片对应的姿态矩阵信息15个和2个深度值范围信息存在这个里。.ravel()函数被调用，将选择的数据拉平为一维数组
    save_arr = np.array(save_arr) #（28，17）
    
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr) # 最后将这个内容保存在npy文件里。也就是说npy里17个参数为一组对应一张图片的姿态矩阵和两个深度值范围
            



def minify_v0(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    def downsample(imgs, f):
        sh = list(imgs.shape)
        sh = sh[:-3] + [sh[-3]//f, f, sh[-2]//f, f, sh[-1]]
        imgs = np.reshape(imgs, sh)
        imgs = np.mean(imgs, (-2, -4))
        return imgs
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgs = np.stack([imageio.imread(img)/255. for img in imgs], 0)
    
    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)
        
        if isinstance(r, int):
            imgs_down = downsample(imgs, r)
        else:
            imgs_down = skimage.transform.resize(imgs, [imgs.shape[0], r[0], r[1], imgs.shape[-1]],
                                                order=1, mode='constant', cval=0, clip=True, preserve_range=False, 
                                                 anti_aliasing=True, anti_aliasing_sigma=None)
        
        os.makedirs(imgdir)
        for i in range(imgs_down.shape[0]):
            imageio.imwrite(os.path.join(imgdir, 'image{:03d}.png'.format(i)), (255*imgs_down[i]).astype(np.uint8))
            



def minify(basedir, factors=[], resolutions=[]): # 给定了colmap处理的内容所在路径：项目路径。和缩放大小
    needtoload = False
    for r in factors: # 遍历需要的缩放大小
        imgdir = os.path.join(basedir, 'images_{}'.format(r))  # 这个大小对应的图片路径
        if not os.path.exists(imgdir): # 路径不存在就需要加载
            needtoload = True
    for r in resolutions: # 遍历需要的分辨率
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0])) # 这个分辨率对应的图片路径
        if not os.path.exists(imgdir): # 路径不存在就需要加载
            needtoload = True
    if not needtoload: # 如果上述需要的路径都存在就可以直接返回
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images') # basedir/images
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))] #basedir/images目录下的每一个文件的路径
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])] # 把上面那个路径中不是以这些结尾的路径剔除
    imgdir_orig = imgdir
    
    wd = os.getcwd() # 获得当前的工作路径

    for r in factors + resolutions: # 遍历所有的缩放和分辨率要求
        if isinstance(r, int): # 是缩放要求时
            name = 'images_{}'.format(r) # 路径的命名
            resizearg = '{}%'.format(int(100./r)) # 命令里面对应的重塑大小的参数
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name) # 希望能读取图片的路径
        if os.path.exists(imgdir): # 路径存在就跳出这一次的循环
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir) # 为这个路径创建对应目录
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True) #把origin里面的文件复制一份到imgdir路径下
        
        ext = imgs[0].split('.')[-1] # 得到路径下第一张图片的后缀名
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)]) # 需要执行的命令就这么写
        print(args)
        os.chdir(imgdir) # 更换工作目录
        check_output(args, shell=True) # 执行这个命令，把ext这个后缀的文件全都按要求resize以下并全部换成png结尾
        os.chdir(wd) # 再更换工作目录
        
        if ext != 'png': # 如果现在处理的这个后缀不是png，那么现在要把这个路径下的ext文件全部移除，避免和已经转换成png的重复
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True) # 执行移除命令
            print('Removed duplicates')
        print('Done')
            
        
        
        
def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    # imgs = [imageio.imread(f, ignoregamma=True)[...,:3]/255. for f in imgfiles]
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    
def gen_poses(basedir, match_type, factors=None): # 输入了colmap处理后的内容所在路径（项目路径），要执行匹配的模式
    
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']] # 需要的三个bin文件名字，camera.bin,images.bin,points3D.bin
    if os.path.exists(os.path.join(basedir, 'sparse/0')): # 如果basedir/sparse/0路径存在，就列举出里面所有的文件名
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else: # 路径不存在，就当作没有一个想要的bin文件都没有
        files_had = []
    if not all([f in files_had for f in files_needed]): # 一旦这个路径下缺少这三个bin文件中的一个都需要重新去运行colmap
        print( 'Need to run COLMAP' )
        run_colmap(basedir, match_type)
    else: # 如果三个bin文件都在，输出提示后可以继续做下面的操作
        print('Don\'t need to run COLMAP')
        
    print( 'Post-colmap')
    
    poses, pts3d, perm = load_colmap_data(basedir) # 得到相机姿态矩阵，所有3D点的信息，匹配上的（也就是和姿态矩阵对应的）图片按名字排序后的索引
    
    save_poses(basedir, poses, pts3d, perm) # 将所有图片对应的姿态信息和点的深度范围存放在npy文件中
    
    if factors is not None: # 默认是没有的，如果有的话
        print( 'Factors:', factors)
        minify(basedir, factors) # 需要生成对应的factor要求的图片集
    
    print( 'Done with imgs2poses' )
    
    return True
    
