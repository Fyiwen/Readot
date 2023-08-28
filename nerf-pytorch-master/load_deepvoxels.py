import os
import numpy as np
import imageio 


def load_dv_data(scene='cube', basedir='/data/deepvoxels', testskip=8):
    

    def parse_intrinsics(filepath, trgt_sidelength, invert_y=False): # 主要从数据集对应文件中读取一些信息，对一些信息可以按照目标大小进行缩放，最后构造一个相机内参矩阵输出，还输出一些读取的别的信息
        # Get camera intrinsics
        with open(filepath, 'r') as file: # 读信息
            f, cx, cy = list(map(float, file.readline().split()))[:3] # 读取文件的第一行，通过空格分隔并转化为浮点数列表。然后将列表中的前三个值分别赋给 f、cx 和 cy。
            grid_barycenter = np.array(list(map(float, file.readline().split()))) # 同样读取文件第二行，得到重心坐标
            near_plane = float(file.readline()) # 读取文件第三行
            scale = float(file.readline()) # 读取文件第四行
            height, width = map(float, file.readline().split()) # 读取文件第五行

            try: # 尝试将文件的第六行转化为整数类型，并将其赋给 world2cam_poses。如果转化出现异常（比如无法转化为整数），则跳过这个步骤。这是为了检查是否存在额外的世界到相机的变换矩阵数量。
                world2cam_poses = int(file.readline())
            except ValueError:
                world2cam_poses = None

        if world2cam_poses is None:
            world2cam_poses = False

        world2cam_poses = bool(world2cam_poses) # 如果有就转换成布尔值

        print(cx,cy,f,height,width)
        # 将cx，cy和f按照目标缩放一下
        cx = cx / width * trgt_sidelength # trgt_sidelength是目标成像一个边的大小
        cy = cy / height * trgt_sidelength
        f = trgt_sidelength / height * f

        fx = f
        if invert_y:
            fy = -f
        else:
            fy = f
        #
        # 构建相机的内参矩阵
        full_intrinsic = np.array([[fx, 0., cx, 0.],
                                   [0., fy, cy, 0],
                                   [0., 0, 1, 0],
                                   [0, 0, 0, 1]])

        return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses


    def load_pose(filename): # 用于从路径中读取poses信息
        assert os.path.isfile(filename) # 判断给定的文件路径filename是否是一个已经存在的文件。如果文件不存在，就会触发断言错误
        nums = open(filename).read().split() # 打开文件并读取，将读取到的内容以空格为分隔符拆分成一个字符串列表
        return np.array([float(x) for x in nums]).reshape([4,4]).astype(np.float32) # 将 nums列表中的每个元素转换为浮点数，并且写成4X4矩阵

    # 得到HWf

    H = 512
    W = 512
    deepvoxels_base = '{}/train/{}/'.format(basedir, scene) # eg:路径/data/deepvoxels/train/cube/

    full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses = parse_intrinsics(os.path.join(deepvoxels_base, 'intrinsics.txt'), H) # 给定了路径和目标大小，得到了相应的内参信息
    print(full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses)
    focal = full_intrinsic[0,0]  # 从内参矩阵中提取焦距值
    print(H, W, focal)
    #
    
    def dir2poses(posedir): # 用于得到相机姿态转换坐标系后的表示
        poses = np.stack([load_pose(os.path.join(posedir, f)) for f in sorted(os.listdir(posedir)) if f.endswith('txt')], 0) # 遍历指定目录下所有以.txt结尾的文件，将文件名和目录路径拼接起来，再加载每个文件中的姿态数据
        transf = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1.],
        ]) # 一个转换矩阵，用于对姿态数据进行变换
        poses = poses @ transf # 将每个姿态矩阵都与变换矩阵相乘
        poses = poses[:,:3,:4].astype(np.float32) # 选择前三行和前四列的子数组，即保留旋转和平移部分，并将其转换为 np.float32 类型，这样做是为了丢弃姿态矩阵的最后一行，因为它在此上下文中不起作用
        return poses
    # 加载要用的相机姿态
    posedir = os.path.join(deepvoxels_base, 'pose') # 相机姿态所在路径
    poses = dir2poses(posedir) # 完成相机姿态的转换（更换了相机坐标系），y轴取反，z轴取反
    testposes = dir2poses('{}/test/{}/pose'.format(basedir, scene))
    testposes = testposes[::testskip] # 得到用于测试的相机姿态
    valposes = dir2poses('{}/validation/{}/pose'.format(basedir, scene))
    valposes = valposes[::testskip] # 得到用于验证的相机姿态
    # 加载要用的样本图片
    imgfiles = [f for f in sorted(os.listdir(os.path.join(deepvoxels_base, 'rgb'))) if f.endswith('png')]
    imgs = np.stack([imageio.imread(os.path.join(deepvoxels_base, 'rgb', f))/255. for f in imgfiles], 0).astype(np.float32)

    testimgd = '{}/test/{}/rgb'.format(basedir, scene)
    imgfiles = [f for f in sorted(os.listdir(testimgd)) if f.endswith('png')]
    testimgs = np.stack([imageio.imread(os.path.join(testimgd, f))/255. for f in imgfiles[::testskip]], 0).astype(np.float32)
    
    valimgd = '{}/validation/{}/rgb'.format(basedir, scene)
    imgfiles = [f for f in sorted(os.listdir(valimgd)) if f.endswith('png')]
    valimgs = np.stack([imageio.imread(os.path.join(valimgd, f))/255. for f in imgfiles[::testskip]], 0).astype(np.float32)
    #
    all_imgs = [imgs, valimgs, testimgs] # 得到所有的图像
    counts = [0] + [x.shape[0] for x in all_imgs] # counts=[0,imgs样本几个，valimgs样本个数，testimgs样本个数]
    counts = np.cumsum(counts) # 将返回一个与输入数组大小相同的新数组，其中每个元素表示原始数组中从开头到当前位置（包括当前位置）的所有元素的累积和
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)] # 得到所有对应的样本索引
    
    imgs = np.concatenate(all_imgs, 0) # 所有样本图像
    poses = np.concatenate([poses, valposes, testposes], 0) # 所有对应姿态矩阵
    
    render_poses = testposes #
    
    print(poses.shape, imgs.shape)
    
    return imgs, poses, render_poses, [H,W,focal], i_split


