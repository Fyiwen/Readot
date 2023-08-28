import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([ # 一个用于平移变换的矩阵，这里表示将三维坐标点在 z 轴方向上进行平移t。
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([ # 一个含有3X3的旋转矩阵，内容表示绕x轴旋转（站在x轴正向顶端看是逆时针转phi）（可以看成x不动，yz轴转）
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([ # 一样包含绕y轴旋转的旋转矩阵（站在y轴正向顶端看是顺时针转theta）
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):  # 输出的c2w矩阵可以将三维坐标从相机坐标系平移到世界坐标系中以原点为中心的球面上。phi为方位角，theta为俯仰角，radius是距离球心的距离，这样可以唯一确定一个球面点
    c2w = trans_t(radius)  # 赋予c2w平移功能，平移距离等于 radius
    c2w = rot_phi(phi/180.*np.pi) @ c2w  # 赋予c2w绕x轴旋转功能
    c2w = rot_theta(theta/180.*np.pi) @ c2w  # 赋予c2w绕y轴旋转功能
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w # 在前面变换基础上，将坐标点在y和z轴方向上进行了交换，并反转了x轴的方向
    return c2w


def load_LINEMOD_data(basedir, half_res=False, testskip=1):  # 用于加载LINEMOD数据集，返回加载的图像、姿态矩阵、渲染姿态、图像尺寸信息、内参矩阵、划分索引以及近截面和远截面
    splits = ['train', 'val', 'test']  # 定义了数据集的划分，包括训练集、验证集和测试集
    metas = {}
    for s in splits: # 遍历每个不同作用的数据集，完成加载
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp: # 打开对应的transform信息文件，eg：basedir/transforms_train.json
            metas[s] = json.load(fp) # 将文件内容解析为Json格式存在对应metas里

    all_imgs = [] # 用于存储所有图像
    all_poses = [] # 用于存储所有姿态矩阵。
    counts = [0]  # 记录每个划分中图像数量
    for s in splits:  # 遍历每个不同作用的数据集，完成所有采样图像和他们对应姿态矩阵的收集
        meta = metas[s]
        imgs = []  # 存储当前划分的图像
        poses = []  # 存储当前划分的姿态矩阵
        if s=='train' or testskip==0: # 如果现在在处理训练集或者采样间隔定为0
            skip = 1  # 图片的采样间隔为1
        else:
            skip = testskip # 否则采样间隔为testskip
            
        for idx_test, frame in enumerate(meta['frames'][::skip]): # 按照采样间隔，遍历当前所在划分的每帧图像信息
            fname = frame['file_path'] # 当前帧所在的文件路径
            if s == 'test': # 如果现在在处理的是测试集，需要呈现一下帧信息
                print(f"{idx_test}th test frame: {fname}")
            imgs.append(imageio.imread(fname)) # 将读取的对应帧的图像添加到列表中
            poses.append(np.array(frame['transform_matrix'])) # 将读取的对应帧的姿态矩阵添加到列表中
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA) # 将图像数据转换为浮点数并进行归一化，像素值从0~255缩放到0~1之间，0时完全透明
        poses = np.array(poses).astype(np.float32) # 将姿态矩阵转成浮点数
        counts.append(counts[-1] + imgs.shape[0]) # 添加当前划分的图像数量。counts[-1]对应的是当前划分的图像数量记录所在，imgs.shape[0]是当前划分中所有图像的个数
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)] # 关于三个划分中所有样本，生成从0开始的连续的索引值
    
    imgs = np.concatenate(all_imgs, 0)  # 将所有划分的图像和姿态矩阵合并成一个大的数组
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2] # 得到所有图像的高宽
    focal = float(meta['frames'][0]['intrinsic_matrix'][0][0]) # 获取所有图像对应的相机焦距
    K = meta['frames'][0]['intrinsic_matrix'] # 获取所有图像对应的内参矩阵
    print(f"Focal: {focal}")
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)# 按照不同angle生成一组c2w矩阵，为了在合成视图视频时能产生晃来晃去的效果（因为相当于找了很多不同的相机点位）
    
    if half_res: # 一旦给定half_res=TRUE
        H = H//2 # 把图像高宽减半，为了降低图片分辨率
        W = W//2
        focal = focal/2.# 图像被缩放后，相机参数也要做相应改变，是外参不变，焦距要等比例缩放

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3)) # 创建新图像，存在里面
        for i, img in enumerate(imgs): # 遍历每一个原图像
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) # 把老图像缩放后存在里面
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    near = np.floor(min(metas['train']['near'], metas['test']['near']))  # 获取所有样本，离相机中心最近的距离
    far = np.ceil(max(metas['train']['far'], metas['test']['far']))  # 获取获取离相机中心最近的距离
    # 之所以需要这两个，是因为体渲染的时候需要在采样区间内选点，那么就可以用near和far来定义这个区间的范围。因为这两个距离更贴近场景边界，在里面采样更能能保证有效。
    return imgs, poses, render_poses, [H, W, focal], K, i_split, near, far


