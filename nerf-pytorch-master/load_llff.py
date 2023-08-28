import numpy as np
import os, imageio


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original
# 这个文件用于读取LLFF格式的真实数据
def _minify(basedir, factors=[], resolutions=[]): # 函数主要负责创建拥有目标分辨率或大小的数据集
    needtoload = False # 默认不需要加载
    for r in factors: # 判断是否存在一些以给定factors部分命名的路径
        imgdir = os.path.join(basedir, 'images_{}'.format(r)) # 图片所在的路径
        if not os.path.exists(imgdir): # 如果路径不存在，就需要加载数据集
            needtoload = True
    for r in resolutions: # 判断是否存在一些以给定分辨率部分命名的路径
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload: # 如果在上面被标记FALSE，也就是路径存在，就可以直接结束函数了
        return
    # 如果路径不存在还要做下面的操作，产生这些路径，包括建目录和产生里面符合要求的图片
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))] # os.listdir(imgdir)以列表形式呈现这个路径下所有文件名字
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])] # 只保留文件名以 JPG、jpg、png、jpeg 或 PNG 结尾的文件路径。用any就是后面只要有一个元素为真就行
    imgdir_orig = imgdir
    
    wd = os.getcwd() # 获取当前工作目录的路径，并将结果存储在变量 wd 中

    for r in factors + resolutions: # 遍历这两个的组合eg：fa=[2,4],re=[【10，10】,【20，20】,【30，30】],那么r是[2, 4, 【10，10】, 【20，20】, 【30，30】]
        if isinstance(r, int): # 如果r是int型，就代表是factors里面的元素
            name = 'images_{}'.format(r) # 得到文件名
            resizearg = '{}%'.format(100./r) # 计算得到需要缩放的大小
        else:  # 如果不是整型，就代表是resolution里面的元素
            name = 'images_{}x{}'.format(r[1], r[0]) # 得到文件名
            resizearg = '{}x{}'.format(r[1], r[0]) # 得到需要缩放到的分辨率
        imgdir = os.path.join(basedir, name) # 得到当前图片应该在的路径名
        if os.path.exists(imgdir): # 路径存在就直接进行下一轮循环
            continue
        # 路径不存在就继续做
        print('Minifying', r, basedir) # 打印要进行压缩处理的值，或分辨率值和基础目录路径
        
        os.makedirs(imgdir) # 创建路径imgdir对应的目录
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True) # 将imgdir_orig目录下的所有文件复制到imgdir目录中
        
        ext = imgs[0].split('.')[-1] # 获取列表imgs中第一个文件的扩展名，并将结果存储在变量ext中
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)]) #形成一个命令例如args= mogrify -resize 100x100 -format png *.ext
        print(args)
        os.chdir(imgdir) # 将当前工作目录更改为 imgdir
        check_output(args, shell=True) # 执行系统命令，将当前目录下所有ext后缀的图像文件进行调整大小并转换为 png 格式
        os.chdir(wd) # 将当前工作目录恢复为原始工作目录
        
        if ext != 'png': # 上面改完以后，文件扩展名是ext的文件全部删掉，因为上面改大小转格式的时候相当于生成了一个新的备份，现在要把重复的原件删掉
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True) # 执行系统命令，删除imgdir目录中扩展名为ext的文件，这样就会在此路径下只剩符合要求的图片文件
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True): # factor是用于缩放的比例系数，将HxW的图像缩放成(H/factor)x(W/factor)
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy')) # 加载这个路径下的数据（是camlap数据经过llff代码处理后所得），应该读取到的信息是NX17，N是图像数量
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 获取除了最后两列之外的所有列，然后改变形状+调整维度。将前15个参数调整成3*5矩阵。3*3旋转矩阵，一列平移量，还一列HWf.这里poses形状为N*3*5
    bds = poses_arr[:, -2:].transpose([1,0]) # 获取最后两列数据，再改一下维度。最后两个参数表示场景的范围near，far，是该相机视角下，场景点离相机中心最近和最远的距离
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0] # 构建了一个路径，获取此路下所有文件列表，过滤出以 'JPG'、'jpg' 或 'png' 结尾的文件，最后取得排序后的列表中的第一个文件路径
    sh = imageio.imread(img0).shape # 得到第一个图像文件的形状信息
    # 上面这两步主要是为了得到原来图片的尺寸，方便下面的缩放操作
    sfx = ''
    
    if factor is not None: # 如果输入时给出了factor
        sfx = '_{}'.format(factor)  # sfx写成_factor
        _minify(basedir, factors=[factor]) # 用给定的factor形成一个对应的数据集
        factor = factor
    elif height is not None:  # 如果输入时给出了缩放后的高度height
        factor = sh[0] / float(height) # 计算对应的缩放比例（用原高/指定高）
        width = int(sh[1] / factor) # 间接可计算出缩放后的宽度（原宽/缩放比例）
        _minify(basedir, resolutions=[[height, width]])  # 生成拥有对应分辨率的数据集
        sfx = '_{}x{}'.format(width, height) # sfx写成_heightxwidth
    elif width is not None: # 如果输入时给出了缩放后的宽度width
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:  # 什么都没给就假定缩放比例为1
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx) # 对应的数据集文件的路径
    if not os.path.exists(imgdir):# 如果路径不存在，需要打印一下
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')] # 在这个数据集文件的路径下，得到他所有的文件列表，筛选出以jpg、png等结尾的，并用这些文件的路径构建文件路径列表
    if poses.shape[-1] != len(imgfiles):# poses的最后一个维度的大小与imgfiles列表的长度不相等，则输出不匹配的提示信息，并直接返回。因为样本的个数和相机位姿的个数得一样
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return

    sh = imageio.imread(imgfiles[0]).shape # 读取imgfiles列表中的第一个图像文件,并获取形状信息。现在可能代表新数据集中图片形状
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) #poses第一维度的前两个元素、第二维度的第四个元素、所有第三维度的元素（就是H，W）设置为sh前两个元素组成的2x1的数组。因为图片尺寸变了，姿态矩阵里面hwf也得对应缩小
    poses[2, 4, :] = poses[2, 4, :] * 1./factor # 最后将poses的第一维度的第3个元素，第二维度的第5个元素（就是f）乘以 1./factor，把焦距也缩放一下
    
    if not load_imgs:
        return poses, bds
    
    def imread(f): # 判断文件路径是否以 'png' 结尾来选择使用读取时要不要忽略gamma校正
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True) # ignoregamma=True 的作用是在加载图像时忽略 gamma 校正。Gamma 校正是一种对图像亮度进行非线性调整的技术，它可以使图像看起来更加逼真和自然
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles] # 读取imgfiles下每一个文件，并将读取到的图像的前三个维度也就是RGB维度进行归一化。[...,:3]中... 表示在其他维度上取所有的元素，:3 表示在最后一个维度上取前三个元素
    imgs = np.stack(imgs, -1)  # 将归一化后的图像文件连接到一起
    
    print('Loaded image data', imgs.shape, poses[:,-1,0]) # 第一个维度上所有元素，第二个维度上最后一个元素，第三个维度上第一个元素
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x) # x除以他的范数值做归一化

def viewmatrix(z, up, pos): # 输入是相机的Z轴朝向、up轴的朝向(即相机平面朝上的方向Y)、以及相机中心，输出的是c2w矩阵，由旋转矩阵R和平移矩阵T组成
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2)) # 用z轴和y轴的叉乘得到关于x轴方向的信息，因为三轴互相垂直
    vec1 = normalize(np.cross(vec2, vec0)) # 也是用叉乘得到y轴方向信息，因为传入的up(Y)轴是通过一些计算得到的，不一定和Z轴垂直
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m # 这是c2w矩阵，描述相机的位置和朝向

def ptstocam(pts, c2w): # 将三维坐标点从w转换到c的函数
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0] # c2w[:3,:3].T提取了c2w矩阵的旋转部分，并对其进行转置。(pts-c2w[:3,3]提取了c2w矩阵的平移部分，并对其与pts进行相减操作。在上述结果的最后一个轴上增加一个新的维度（为了确保矩阵乘法的维度匹配），矩阵乘法实现了三维坐标点的转换，最后去除结果中的最后一个维度，将结果转换为一维数组。
    return tt # 示转换后的坐标点在摄像机坐标系中的位置

def poses_avg(poses): # 输入是多个相机的位姿，输出多个相机的平均位姿（包括位置和朝向）。这里poses形状为N*3*5

    hwf = poses[0, :3, -1:] # 取第一个相机位姿的前三行和最后一列，也就是得到形状为3*1对应图像的hwf。

    center = poses[:, :3, 3].mean(0) # 对多个相机的中心进行求均值得到center
    vec2 = normalize(poses[:, :3, 2].sum(0)) # 对所有相机的Z轴求平均得到vec2向量（方向向量相加其实等效于平均方向向量）。
    up = poses[:, :3, 1].sum(0) # 对所有的相机的Y轴求平均得到up向量
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1) #将得到的vec2, up和center输入到viewmatrix()函数就可以得到平均的相机位姿4*4,然后再连接3*1得到4*5的新c2w矩阵
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):  # 为了生成一段螺旋式的相机轨迹用于新视角的合成（具体的：相机绕着一个轴旋转，其中相机始终注视着一个焦点，相机的up轴保持不变）。zdelta: 相机在焦点和相机位置之间的初始距离。zrate: 控制相机在旋转过程中沿着 Z 轴移动的速度。rots: 控制相机绕着焦点旋转的圈数。
    render_poses = [] # 用于储存得到的相机位姿
    rads = np.array(list(rads) + [1.]) # rads转换为一个带有额外元素1.0的NumPy数组，多了1可能是为了长度和其他数组对齐。rads应该是相机绕焦点旋转的半径值
    hwf = c2w[:,4:5] # 将c2w的第五列提取出来作为hwf数组
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]: # 每一迭代生成一个新的相机位置theta，这个的取值来自从0-2pi*rot之间均匀分布的值，除了最后一个
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) # 得到当前迭代的相机中心在世界坐标系的位置，这一步又实现了旋转，旋转时沿z轴移动，而且旋转时的半径也由rads提供可能发生改变
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))  # c-np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])得到焦点在世界坐标系的位置(c-(-fz1+c1,-fz2+c2,-fz3+c3))得到（-fz1,-fz2,-fz3），然后归一化，相当于单位化,得到的z是相机z轴在世界坐标系的朝向
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))  # 用viewmatrix(z, up, c)构造当前相机的矩阵，然后再和hwf拼接得到4*5矩阵
    return render_poses
 #上面这个因为假设所有相机都朝向某一个方向，所以能合成所谓的faceforward场景


def recenter_poses(poses): # 输入N个相机位姿，返回N个相机位姿。为的是中心化每个相机位姿

    poses_ = poses+0 # 相当于创建了poses数组的一个副本赋值给poses_，之所以需要他是因为我们想只更改想姿态矩阵中三行四列的旋转加平移信息，而其他信息不变
    bottom = np.reshape([0,0,0,1.], [1,4]) # 将一个列表重塑成1*4矩阵，用于将矩阵补成其次坐标的形式
    c2w = poses_avg(poses) # 得到多个输入相机的平均位姿c2w是3*5的，前三列对应xyz，第四列对应相机中心，第五列对应hwf
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)  # 把平均位姿表示的原来的三行四列改成齐次坐标形式的四行四列
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1]) # 重塑后的三维bottom再三个维度上分别重复poses第一个维度的大小N次,1次,1次，形状为N*1*4
    poses = np.concatenate([poses[:,:3,:4], bottom], -2) # 把所有姿态矩阵中原来的三行四列改成齐次坐标形式的四行四列

    poses = np.linalg.inv(c2w) @ poses #用这个平均位姿c2w的逆矩阵左乘到输入的相机位姿上就完成了归一化。因为同一个旋转平移变换矩阵左乘所有的相机位姿是对所有的相机位姿做一个全局的旋转平移变换
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses
# 这些相机会被变到什么样的一个位置？如果把平均位姿的逆c2w^-1左乘平均相机位姿c2w，返回的相机位姿中旋转矩阵为单位矩阵，平移量为零向量。也就是变换后的平均相机位姿的位置处在世界坐标系的原点，XYZ轴朝向和世界坐标系的向一致。

#####################


def spherify_poses(poses, bds):#这个函数可以用于"球面化"相机分布并返回一个环绕的相机轨迹用于新视角合成，以得到相机围绕着一个物体拍摄的360度场景

    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)  # 用于将输入的3x4矩阵p扩展为齐次表示的4x4矩阵，令单位矩阵为最后一行，重塑后重复，再和p做连接
    
    rays_d = poses[:,:3,2:3] # 提取出所有相机的方向向量（也就是中心射线方向），形状（n，3，1）
    rays_o = poses[:,:3,3:4] # 提取出所有相机的中心坐标（也就是中心射线原点），形状（n，3，1）

    def min_line_dist(rays_o, rays_d): # 为找到离所有相机中心射线距离之和最短的点（可以先简单理解成场景的中心位置）
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1]) # 一个投影矩阵
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d) # 将得到的最优点位置设为场景的中心

    # 以场景中心为原点写出一个相机姿态矩阵，用这个矩阵把之前相机坐标系下的姿态矩阵，转成世界坐标系下的姿态矩阵表示，并且在转换的过程中将所有相机z轴的平均方向转到和世界坐标系的z轴相同
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)  # 计算所有相机中心坐标与场景中心位置的差的平均值向量，作为up轴

    vec0 = normalize(up) # 计算出新坐标系的三个轴的单位向量表示
    vec1 = normalize(np.cross([.1,.2,.3], vec0)) # 只要选一个与vec0不共线的向量，就可以求到与vec0垂直的向量，[1,2,3]的选择比较常见
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center # 将场景中心位置设为新的相机坐标系的原点
    c2w = np.stack([vec1, vec2, vec0, pos], 1) # 形成新的c2w矩阵

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4]) # 用新的c2w完成将相机姿态从相机坐标系转换为世界坐标系
    #接下来开始求用于渲染的相机姿态
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1))) # 计算在新的相机中心位置离世界坐标系中心的平均距离
    # 下面利用rad将相机的位置缩放到单位圆内
    sc = 1./rad # 缩放因子
    poses_reset[:,:3,3] *= sc # 相机位置缩放
    bds *= sc # 场景边界缩放
    rad *= sc # 半径缩放
    #正式开始求渲染用的相机姿态
    centroid = np.mean(poses_reset[:,:3,3], 0) # 计算新表示下的相机位置们的中心位置
    zh = centroid[2] # 拿出中心位置的z坐标赋值
    radcircle = np.sqrt(rad**2-zh**2)  # 计算出一个半径值。下面会按照这个固定的半径，生成新的相机中心
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120): # 等间距数组表示的角度值，用于确定各个相机在圆周上的位置

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh]) # 得到用作渲染的相机光心位置，光心在z轴上的对应位置是固定的
        up = np.array([0,0,-1.]) # 表示了相机的朝向，这里朝-z轴方向

        vec2 = normalize(camorigin) # 以每个相机光心向量作为z轴方向
        vec0 = normalize(np.cross(vec2, up)) # 计算得到x轴方向
        vec1 = normalize(np.cross(vec2, vec0)) # 计算得到y轴方向
        pos = camorigin # 相当于平移量，也是相机中心位置
        p = np.stack([vec0, vec1, vec2, pos], 1) # 形成相机姿态矩阵

        new_poses.append(p) # 得到新的相机轨道上，所有相机的位置

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1) # 得到用于渲染的姿态
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1) # 世界坐标系下的相机姿态
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    

    poses, bds, imgs = _load_data(basedir, factor=factor)  # 从数据集文件中读取到的相机姿态矩阵，场景边界和对应图像
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # 实现把旋转矩阵的第一列（表示对X轴的操作）和第二列（表示对y轴的操作）互换，并且对第二列（Y轴）做了一个反向，目的是将LLFF的相机坐标系变成NeRF的相机坐标系

    poses = np.moveaxis(poses, -1, 0).astype(np.float32) # 将最后一个维度-1移动到第一个维度0，其他维度保持不变
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor) # 计算尺度因子
    poses[:,:3,3] *= sc # 对相机的姿态矩阵中的平移分量进行缩放
    bds *= sc # 对场景边界缩放
    
    if recenter:  # 如果为true，就中心化每个相机的位姿
        poses = recenter_poses(poses)
        
    if spherify:  # 如果为true，就对姿态矩阵进行球面化的操作
        poses, render_poses, bds = spherify_poses(poses, bds)

    else: # 否则就要经过下面的操作生成渲染路径上的相机姿态
        
        c2w = poses_avg(poses) # 计算所有相机姿态的平均位姿
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0)) # 对所有相机姿态矩阵中有关y轴方向的向量取和，相当于求平均，在归一化得到up轴

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5. # 场景最小边界乘0.9作为近焦点深度，最远边界乘5作为远焦点深度
        dt = .75  # 用于下面加权的参数
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth)) # 用加权和，计算出平均深度
        focal = mean_dz  # 用平均深度作为焦距

        # Get radii for spiral path
        shrink_factor = .8 # 缩小因子，用于控制螺旋路径的大小
        zdelta = close_depth * .2 # 用近焦点深度求到的值*0.2，作为沿着Z轴的位移量，用于控制相机在螺旋路径上的垂直移动。
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T # 提取所有相机姿态矩阵的平移部分
        rads = np.percentile(np.abs(tt), 90, 0) # 计算这个新数组在列方向上的百分数为90的结果。即找到一个数能使tt中90%值包含在这个数范围内。计算结果每列对应一个，这个结果的计算依靠loc = 1 + (n - 1) * p ，其中n为排序后数的个数，p为百分位，num是结果 = a[loc整数部分 - 1] + (a[loc整数部分] - a[loc整数部分 - 1]) * loc小数部分，a就是排序后的新数组
        c2w_path = c2w # 相机的平均位姿
        N_views = 120  # 螺旋路径中观察点数量
        N_rots = 2  # 螺旋路径中旋转次数
        if path_zflat: # 一旦为true
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1 # 设置一个沿着Z轴的偏移量，用于将相机沿着Z轴向下移动。
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2] # 将原来的相机姿态矩阵在Z轴方向上进行偏移。
            rads[2] = 0. # 将螺旋路径中的Z轴方向上的半径设为0，使相机沿着Z轴不发生旋转。
            N_rots = 1 # 旋转次数改
            N_views/=2 # 观察点个数也改

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views) # 产生旋转路径上的相机姿态矩阵们
        
    # 得到最终用于渲染的相机姿态
    render_poses = np.array(render_poses).astype(np.float32)
    # 把和平均位姿最近的那个相机姿态保留下来作为测试集中的一员
    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1) # 计算每个相机姿态矩阵对应的相机中心和平均姿态矩阵中表示的相机中心之间的距离
    i_test = np.argmin(dists) # 找到距离最近的那个相机姿态，在所有相机姿态中的索引值
    print('HOLDOUT view is', i_test)
    # 图像和对应姿态矩阵
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test # 这个poses是3*5就是还加了一列hwf



