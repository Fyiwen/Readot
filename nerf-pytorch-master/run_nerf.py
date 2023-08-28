# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk): # fn是一个nerf网络，chunk是一个批次的大小
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None: # 如果每个批次大小，直接返回fn
        return fn
    def ret(inputs):# 如果给了就会返回一个ret函数，这个函数把inputs[0:chunk]，inputs[chunk:2chunk]..把input分批次放入fn中，然后再把每个的结果连接起来
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64): #给定输入点x（inputs）和对应方向d（viewdirs）后，经过分别的位置编码后，送入预测点的颜色和体积密度的nerf网络fn，这个网络处理时按照netchunk的批次大小分别处理。embed_fn是做点x的位置编码的嵌入函数, embeddirs_fn是做方向d的位置编码的嵌入函数, netchunk是把输入分开的批次大小
    """Prepares inputs and applies network 'fn'.
    """
    # input=[256,64,3],256根射线上，每个射线64个采样点，每个采样点3d位置
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) #[256*64,3] 将输入点x的形状做出改变
    embedded = embed_fn(inputs_flat) #[256*64,60+3] 输入点x进入嵌入函数，得到位置编码后的结果

    if viewdirs is not None: # 如果给了方向d，也需要对他的形状加以改变后，完成位置编码，并且得到一个变量是x和d位置编码后连接起来的结果
        input_dirs = viewdirs[:,None].expand(inputs.shape) # [256,64,3]先在最后一个维度扩展维度，然后再把他扩展成和input一样的形状
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]]) #[256*64,3] 完成位置编码前的形状改变
        embedded_dirs = embeddirs_fn(input_dirs_flat) # [16384,24+3]进行d的位置编码
        embedded = torch.cat([embedded, embedded_dirs], -1) #[16384,60+3+24+3] 把x和d的编码结果连接

    outputs_flat = batchify(fn, netchunk)(embedded) #[16384,4] 把输入embedded，分成netchunk的批次大小，分别放到fn网络中，得到分别的输出连接起来的结果
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]) #[256,64,4] 输出形状调整一下
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs): # 输入射线信息，批次大小和其他一些参数
    """Render rays in smaller minibatches to avoid OOM.为了避免内存不足的问题，使用更小批次渲染射线
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk): # i=[0,chunk,2chunk.3chunk....]
        ret = render_rays(rays_flat[i:i+chunk], **kwargs) # 每次提取chunk个射线信息，送入函数中，得到对应的ret，ret里面是个字典，字典里每个参数对应的是这些射线处理后得到的颜色图等等图
        for k in ret:  # 遍历ret中每一个字典，ret[k],k完整情况下包含rgb_map,disp_map,acc_map,raw,rgb0,disp0,acc0,z_stdrgb_map,disp_map,acc_map,raw,rgb0,disp0,acc0,z_std
            if k not in all_ret:  # 如果当前all_ret中还没有存储过这个k参数相关的信息，就新建一个空白列表用于存放
                all_ret[k] = []
            all_ret[k].append(ret[k]) # 把这批射线的ret里面的k信息存到all_ret[k]里面

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret} # 所有射线的ret[k]信息都存在了all_ret【k】里面
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. 图片的像素高度.
      W: int. 图片的像素宽度
      focal: float. 针孔摄像机的焦距
      chunk: int. 批量大小，同时处理的最大光线数。
        用于控制最大内存使用量。不影响最终结果。
      rays: array of shape [2, batch_size, 3]. 存储一个批次中每个样本射线的起点和方向，rays在train函数后面的训练部分被赋值，表示那些用来训练的射线
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, 表示ndc坐标系下的相机原点和方向.
      near: float or array of shape [batch_size]. 一个射线最近的距离
      far: float or array of shape [batch_size]. 一个射线最远的距离。采样点只能在这个范围之内
      use_viewdirs: bool. If True, 在模型中使用空间中的点的视图方向.
      c2w_staticcam: array of shape [3, 4]. If not None, 将此转换矩阵用于相机，同时对观察方向使用其他C2W参数。
    Returns:
      rgb_map: [batch_size, 3]. 预测的射线的颜色值
      disp_map: [batch_size]. Disparity map. Inverse of depth.视差图。深度的反比。视差图是用于描述双目视觉系统中的图像深度差异的一种表示方式。在双目视觉中，由于每个眼睛的位置不同，同一个物体在左右眼的图像中会有微小的位置差异，这种位置差异称为视差。视差图则是将这些视差值在图像中以灰度或彩色的方式进行可视化。
                                                                              深度是指物体离相机的距离，深度值表示了场景中不同物体的远近关系。而视差与深度有一种反比的关系，也就是说，如果物体离相机更远，那么视差值会更小；反之，如果物体离相机更近，则视差值会更大。因此，通过计算视差值并进行适当的转换，我们可以得到深度信息
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray. 沿着一条射线积累的透明度。所有射线组成的透明度图
      extras: dict with everything returned by render_rays().字典，是由render_rays()函数返回的，包含了渲染射线操作的各种结果
    """
    if c2w is not None: # 如果给了相机姿态矩阵c2w（一个），那么就可以通过下面这个函数得到这个视角下的所有射线信息，每个射线用原点和方向表示
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w) # 这里c2w只包含一个相机姿态，得到这个视角下的射线原点和方向
    else: # 否则会提供rays变量，里面是直接的射线信息。这个rays在train函数后面的训练中会被复制，表示当前训练批次中，需要得到结果的射线，通过更下面的操作得到这些射线的预测信息，用于和target比较
        # use provided ray batch
        rays_o, rays_d = rays  # 都是[256,3]

    if use_viewdirs: # 如果这个为true表示，要使用视角信息，就要做下面的操作
        # provide ray directions as input
        viewdirs = rays_d # 把射线方向给这个变量
        if c2w_staticcam is not None: # 如果这个矩阵存在，就用这个c2w_staticcam求出对应射线信息
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True) # 将射线方向进行归一化[256,3]
        viewdirs = torch.reshape(viewdirs, [-1,3]).float() # 重塑形状[256,3]
    # 到这边为止viewdirs一定存的是用c2w得到的射线方向，而rays_d可能存的是用c2w得到的射线方向，也有可能是c2w_staticcam得到的
    sh = rays_d.shape # [256, 3]
    if ndc: # 如果为true，就是需要转成ndc坐标系
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d) # 将相机坐标系下的射线信息转换到ndc坐标系下

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float() # 该形状适应下面操作[256,3]
    rays_d = torch.reshape(rays_d, [-1,3]).float() # [256,3]

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])  # 复制一点内容方便下面的操作(256,1)
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # rays里面以这样的格式存储每一条射线(256,8)
    if use_viewdirs:  # 如果使用视图信息，需要连接两个不同表示的射线变量（产生时用的c2w不一样）
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs) # 输入所有射线的信息，这些射线即将被分批次处理的批次大小，最终得到所有射线的ret[k]信息都存在了all_ret【k】里面，所以函数返回allret
    for k in all_ret: # 遍历all_ret里面每一个all_ret[k]对应存放的信息，包括rgb_map,disp_map,acc_map,raw,rgb0,disp0,acc0,z_std
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:]) #[256,3] 定义了一个形状
        all_ret[k] = torch.reshape(all_ret[k], k_sh) # 改变一下all_ret[k]的形状
   # 现在all_ret[rgb_map]=[256,3],其他几个256，[raw]=[256,192,4],但是调试看到，没改形状之前也这样
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract] # 把all_ret[rgb_map],all_ret[disp_map],all_ret[acc_map]这三类图的信息存在ret_list里面
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract} # 其他类的图的信息存放在ret_dict里面
    return ret_list + [ret_dict]  # 返回的是能从射线中得到的一系列信息【all_ret[rgb_map]，all_ret[disp_map]，all_ret[acc_map]】+【all_ret里面其他的一些图和信息】


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):  # 给定用于视图合成的相机渲染路径（生成的），hwf，K，批次大小，测试时要用的渲染参数，供渲染出来的图片比较的测试集真实图（注意如果用测试集来渲染就有，用生成的渲染路径渲染视图就没有），渲染出来的图的保存路径，下采样因子以加快渲染速度，设置4或8以快速预览

    H, W, focal = hwf

    if render_factor!=0: # 如果给了下采样因子，hwf都会进行缩放，这样渲染出来的视图分辨率就会低了
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []  # 会存所有相机姿态分别对应得到的rgb图（一个姿态矩阵会得到很多关于他的射线，从这些射线上获取信息，会得到rgb图）
    disps = []  # 会存所有相机姿态分别对应得到的视差图（类似得到的视差图）

    t = time.time()  # 开始渲染的时间
    for i, c2w in enumerate(tqdm(render_poses)):  # 遍历每一个需要渲染视图出来的相机姿态。这里tqdm()函数接受一个可迭代对象作为参数，并返回一个包装后的可迭代对象，在每次迭代时会显示一个进度条，可以在循环遍历时以进度条的形式显示迭代的进度。
        print(i, time.time() - t) # 打印出当前的索引值i和上次循环的执行时间
        t = time.time() # 现在的时间
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs) # 输入图像高宽内参矩阵，批次大小，当前相机姿态矩阵3*4，以及渲染所需的一些参数，返回的是能从相机姿态对应的射线中得到的一系列信息【all_ret[rgb_map]，all_ret[disp_map]，all_ret[acc_map]】+【all_ret里面其他的一些图和信息】
        rgbs.append(rgb.cpu().numpy())  # 把当前相机姿态对应得到的颜色图存放到rgb这个变量里面
        disps.append(disp.cpu().numpy()) # 把当前相机姿态对应得到的视差图存放到disp这个变量里面
        if i==0: # 如果是第一次循环，输出rgb图和视差图的形状信息
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None: # 如果给出了渲染出的图片应该存放的路径
            rgb8 = to8b(rgbs[-1]) # 将输入的当前循环的rgb图转换为取值范围在0到255之间的无符号8位整数（uint8）类型（因为rgbs[-1]对应rgb这个列表中的最后一个元素，也就是当前循环中刚刚加进去的那个rgb图）
            filename = os.path.join(savedir, '{:03d}.png'.format(i)) # 给出了文件名字
            imageio.imwrite(filename, rgb8) # 将名为rgb8的图像数据写入filename对应的文件中


    rgbs = np.stack(rgbs, 0)  # 所有相机姿态能得到的颜色图
    disps = np.stack(disps, 0)  # 所有相机姿态能得到的视差图

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)  # 给定位置编码的最大频率的L=10（3D位置），得到用于位置编码的嵌入函数和嵌入器对象的输出维度.
    # embed_fn是点x的位置编码函数 input_ch 是编码后的点x输入nerf网络的通道数
    input_ch_views = 0 # embeddirs_fn是方向d的位置编码函数input_ch_views是编码后的方向d输入nerf网络的通道数
    embeddirs_fn = None
    if args.use_viewdirs: # 如果这个为true，根据给定位置编码的最大频率的L=4（2D方向）就生成另一个嵌入函数和嵌入器对象的输出维度
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4 # 如果在每条射线上额外采样点的个数（这些点只用于细网络）>0，输出维度就是5，否则为4
    skips = [4] # 额外需要处理的一个线性层索引
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)  # 创建了一个nerf模型，“粗网络”。
    grad_vars = list(model.parameters()) # 将这个“粗”模型的参数添加到梯度变量列表中

    model_fine = None
    if args.N_importance > 0: # 如果有只能细网络用的额外采样点，那么就要构建细网络了
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device) # 创建了另一个nerf模型，“细模型”
        grad_vars += list(model_fine.parameters())  # 把这个“细”模型的参数也添加到梯度变量列表中

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk) # 定义了一个匿名函数，可以在只给出inputs（x点）, viewdirs（方向d）, network_fn（nerf网络）的情况下执行run_network函数得到这些输入经过网络后的输出

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999)) # 优化器，里面定义了两个衰减系数。在通常情况下，β1接近于1，β2接近于0.999，以保持对过去梯度信息的较长记忆

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None': # 如果指定了路径，就用这个路径
        ckpts = [args.ft_path] # 这个路径下是训练时记录下来的权重npy文件用于重新加载出训练好的网络，eg:./logs\\blenser_paper_lego\\002000.tar
    else: # 没给就用这个路径
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f] # 没给这个参数，就自己定义这个路径

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload: # 存在检查点文件，且没有这个命令：不从保存的ckpt文件中重新加载权重，就执行下面的操作
        ckpt_path = ckpts[-1] # 得到最新的检查点文件路径，设置为ckpt列表中的最后一个元素
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path) # 加载检查点文件并且内容保存在ckpt中

        start = ckpt['global_step'] # 从检查点文件中获取全局步数，这个值是训练过程中记录的训练步数或迭代次数。
        optimizer.load_state_dict(ckpt['optimizer_state_dict']) # 加载优化器的状态字典，将之前保存在检查点中的优化器状态恢复到当前的优化器对象

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict']) # 加载模型的状态字典，将之前保存在检查点中的模型状态恢复到当前的模型对象。
        if model_fine is not None: # 如果“细网络”在，还要加载他的状态字典
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {  # 训练时，体渲染要用的参数
        'network_query_fn' : network_query_fn,  # 用于得到nerf网络的输出的函数
        'perturb' : args.perturb,  # 非0，就表示会进行扰动 ##################################################
        'N_importance' : args.N_importance,  # 用于细网络的每条射线上的额外采样点个数
        'network_fine' : model_fine,  # 细网络模型
        'N_samples' : args.N_samples,  # 沿着同一个射线的不同粗采样次数
        'network_fn' : model,  # 粗网络模型
        'use_viewdirs' : args.use_viewdirs, # 决定使用带方向的5D输入还是3D
        'white_bkgd' : args.white_bkgd,  # 设置在白色背景上渲染合成数据
        'raw_noise_std' : args.raw_noise_std, # 添加了标准设备噪声以正则化sigma_a输出（体积密度） ##############
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:  # 如果现在使用的不是llff数据集，或者后面这个参数为true
        print('Not ndc!')
        render_kwargs_train['ndc'] = False  # 就在要用的参数列表里添加一个'ndc' = False
        render_kwargs_train['lindisp'] = args.lindisp  # 再添加一个'lindisp' = args.lindisp，如果为true，表示以反深度而不是深度进行线性采样。

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train} #同样把train要用的这些参数复制给test一份
    render_kwargs_test['perturb'] = False  # test中这个参数直接设为false
    render_kwargs_test['raw_noise_std'] = 0.  # 这个参数设为0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer # 返回训练要用的参数列表，测试要用的参数列表，训练时的迭代次数，要优化的梯度变量们，优化器


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False): # 采样点送入粗网络得到的结果，射线上那些采样点的深度值，射线方向，还有三个参数
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.粗模型的预测结果，对那些采样点
        z_vals: [num_rays, num_samples along ray]. Integration time.射线上那些采样点的深度值
        rays_d: [num_rays, 3]. Direction of each ray. # 每条射线的方向
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists) # 这个匿名函数给定粗网络的输出，采样点间隔距离和干扰函数，从公式上看是计算论文中射线颜色C_hat(r)的一部分，也可以看作是alpha合成方法中alpha值的写法公式。真正调用的时候直接用体积密度乘dist，我记得论文里有提过这个操作对网络的输出结果进行了扰动，使得渲染效果更好了

    dists = z_vals[...,1:] - z_vals[...,:-1] #[256,64] dist存的射线上是每两个采样点之间的间隔距离
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays=256, N_samples=64] # 将一个具有相同形状的大数值（1e10）添加到 dists 张量的末尾，扩展了最后一个维度。这个操作的目的是在 dists 张量中的每个样本序列的末尾添加一个很大的距离值，以确保在射线与物体没有相交时，距离值会超过所有可能的相交距离，从而指示没有相交发生

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  #[256,64] 将dists乘以rays_d的范数。就是根据光线方向的长度对距离值进行缩放。因为光线方向不一样，长度也不一样，这里相当于把所有射线的长度统一下

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3] 用sigmoid对粗网络输出的rgb颜色进行处理，因为sigmoid将输入值映射到[0, 1]的范围内，所以相当于对颜色进行了归一化
    noise = 0.
    if raw_noise_std > 0.:  # 如果噪声标准差大于0表示需要添加噪声
        noise = torch.randn(raw[...,3].shape) * raw_noise_std # 首先生成了与raw中颜色信息形状相同的随机数，再乘上噪声标准差，得到每个颜色位置应该赋予的噪声

        # Overwrite randomly sampled data if pytest
        if pytest: # 如果给定了pytest，就表示要用下面的固定随机数产生噪声，而不是上面那种不固定的
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    # 这边给颜色加了噪声
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples] 给raw中每个颜色信息加上噪声，然后再用raw和dist作为输入执行上面那个匿名函数
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1] #[256,64] 计算权重w，根据论文里那个wi的公式
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3] 使用上面算到的权重对对应点的颜色值加权求和，得到每一条射线的颜色

    depth_map = torch.sum(weights * z_vals, -1)  #[256] 用权重值对采样点深度值加权求和，得到每个射线穿过物体的深度值，然后组成图
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)) #[256] 视差图，每个射线的视差形成的图
    acc_map = torch.sum(weights, -1) #[256] 每条射线积累的采样点的权重和形成的图。

    if white_bkgd: # 如果需要白色背景，就要对下面的颜色图做这个操作
        rgb_map = rgb_map + (1.-acc_map[...,None])  #[256,3] 因为rgb颜色从0-1，1的话就是白色，这边在原来rgb的基础上+1-acc，acc小的地方表示那条射线上的点都不太重要（或者说是对渲染的这个物体不太重要），最有可能是背景

    return rgb_map, disp_map, acc_map, weights, depth_map  # 返回每一个射线的颜色组成的颜色图，视差图，每条射线上采样点的权重和组成的图，每个射线上采样点的颜色的权重，每个射线穿过物体的深度值组成的图


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. 一批次的射线们，里面包含沿着一个射线采样所有必要的信息。包含：原点，方向，最近的射线距离，最远的（在射线上采样只能在这两个范围内取点）
        and unit-magnitude viewing direction.单位长度的射线方向（因为这里可能会有两个射线方向，一个是用c2w得到的，还一个用c2w_s...得到）
      network_fn: function.nerf模型，预测空间中每个点的rgb和体积密度。“粗网络”
      network_query_fn: 一个函数用于把输入送入nerf网络，得到输出
      N_samples: int. 沿着一根射线需要完成的采样次数。
      retraw: bool. If True, include model's raw, unprocessed predictions.如果为true，包含模型的原始、未经处理的预测结果
      lindisp: bool. If True, 采样时使用逆深度（inverse depth）线性采样，而不是直接采样深度（depth）。深度是指从观察者到场景中各个点的距离，逆深度是深度的倒数，即 1/depth。逆深度的一个重要特性是，当物体离观察者较远时，逆深度值很小，而当物体靠近观察者时，逆深度值很大。当我们在逆深度上均匀采样时，实际上是在物体距离上进行非线性采样，使得较远的物体获得更少的采样点，而较近的物体则获得较多的采样点
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.大于0，就表示需要扰动本来均匀分布的采样点
      N_importance: int. 沿着每个射线额外的采样次数
        这些采样点只能用于细网络.
      network_fine: 和粗网络规格一样的细网络
      white_bkgd: bool. If True, 假设一个白色背景.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.每个样本的沿射线距离的标准差
    """
    N_rays = ray_batch.shape[0]  # [256]得到这一批次中射线的总个数
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each 得到这一批次中所有射线的原点和方向
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None  #[256,3] 根据ray_batch最后一维的形状可以判断里面有没有包含 viewdirs这个方向信息，有的话提取出来赋给这个变量
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) # [256,1,2]改换形状方便从射线信息中提取near和far的信息
    near, far = bounds[...,0], bounds[...,1] # [256,1] # 等会儿在near和far之间采样

    t_vals = torch.linspace(0., 1., steps=N_samples) #[64] 根据需要采样点的个数N_samples，得到了在0-1之间均匀分布的t值，方便后面的采样操作
    if not lindisp: # 如果不用逆深度线性采样，那就是沿着深度从near位置开始到far位置之间均匀采样
        z_vals = near * (1.-t_vals) + far * (t_vals) # [256,64]里面存的是每个采样点的z坐标（在射线上的深度值）
    else: # 使用逆深度线性采样
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals)) # 得到每个采样点的z坐标（深度值更准确）

    z_vals = z_vals.expand([N_rays, N_samples]) #[256,64] 将z_vals的维度扩展到[N_rays, N_samples]每个射线对应采样点的深度值

    if perturb > 0.: # 如果要扰动采样点，就用下面的操作得到扰动后的采样点深度值z_vals
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1]) #[256,63] 得到相邻深度值之间的中间值mid（也就是两个相邻的采样点之间的中间点的z值）
        upper = torch.cat([mids, z_vals[...,-1:]], -1) #[256,64] upper包含所有mid和最后一个深度值（最后一个采样点）
        lower = torch.cat([z_vals[...,:1], mids], -1) #[256,64] lower包含第一个深度值（第一个采样点）和所有mid
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)  #[256,64] 生成随机采样值，形状和正经的均匀采样z_vals一致。生成的是随机数张量，也在[0, 1)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest: # 如果设置了pytest，就是需要在测试中使用固定的随机数生成器，也就是将上面随机的t_rand覆盖为numpy生成的固定随机数数组，并将其转换为张量格式
            np.random.seed(0) # 这里定了种子0，那么每次都会得到固定的随机数序列
            t_rand = np.random.rand(*list(z_vals.shape))  # 随机数组t_rand 和 z_vals 具有相同的形状，是均匀分布在 [0, 1) 范围内的随机数
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand #[256,64] 得到扰动后的采样点深度值.每个点的深度值来自每两个mid之间的一个随机值

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3] # 用o+tnd得到所有采样点坐标


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn) #[256,64,4] 把这些采样点送入粗网络，得到输出结果
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest) # 输入采样点经过粗网络后的结果，采样点的深度值，射线方向们，还有三个影响操作的参数 返回每一个射线的颜色组成的颜色图，视差图，每条射线上采样点的权重和组成的图，每个射线上采样点的颜色的权重，每个射线穿过物体的深度值组成的图

    if N_importance > 0: # 如果需要沿着射线额外再采样N_importance个点，用于细网络

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map # 把之前用粗网络输出得到的三个图，存在这三个带0的变量里

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])  #[256,63] 表示相邻采样点的深度值之间的中间值mid，相当于得到了新的采样点的深度值
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest) #[256,128] 得到分层采样的结果，也是用深度值表示
        z_samples = z_samples.detach()  # 相当于在计算图中切断了z_samples与之前操作的梯度传播链。不再保留与计算图的连接，也即不再参与梯度计算

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1) #[256,128+64] 将之前粗网络的采样点深度值z_vals和现在根据粗网络的采样点重新采样的点深度值z_samples，连接起来并且排序好放在一起，索引值这里不要
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]计算得到采样点的位置（N_samples + N_importance）个

        run_fn = network_fn if network_fine is None else network_fine # 如果细网络存在，那么run_fn表示细网络那个模型，否则还是指粗网络模型
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn) #[256，128+64，4] 把采样点和射线方向全部送入细网络，得到输出结果raw

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest) # 将输出结果等送入这个函数，得到与结果相关的颜色图、视差图、积累图、权重、深度图
       #[256,3],  [256],   [256],   [256,192],[256]
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}  # ret是和这三个图相关的字典
    if retraw:  # 如果为true，ret就需要包含模型的原始、未经处理的预测结果，也就是ret中增加下面的这一行
        ret['raw'] = raw  # raw里面就是模型的原始、未经处理的预测结果
    if N_importance > 0:  # 如果需要沿着射线额外再采样N_importance个点用于细网络，那么本来ret中三个图是来自细网络输出，ret还要增加下面的粗网络的输出图们
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays] #计算z_samples在最后一个维度上的标准差（也就是所有采样的深度值之间的标准差）

    for k in ret:  # 遍历ret中每一个信息，判断ret[k]里面有没有存在任何一个值是NAN或者无穷大，并且在调试模式下，就可以输出提示信息
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret # 返回这个存满信息的ret，完整情况下有


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, #
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=64*32,  # 太大了改了1024*32
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=64*64,  # 太大了改了1024*64
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')  #

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    # For real scene data, we regularize our network by adding
    # random Gaussian noise with zero mean and unit variance to the output σ values
    # (before passing them through the ReLU) during optimization
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops') # 中心位置才有图像，边上可能都是白色背景。所以一开始先让中心训练
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)') # 除了 face forward 场景以外都不用ndc坐标系
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,  #因为llff的8个场景中数据量不同，所以按照比例设置测试集，llff数据集中每 8张图片片中选择一张作为数据集
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=2000,  # 10000
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()  # 创建一个配置解析器对象
    args = parser.parse_args()  # 解析命令行参数，并将解析结果存储在args变量中

    # Load data
    K = None
    if args.dataset_type == 'llff': # 如果现在使用的是llff数据集
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)  # 加载数据集内容得到，样本图像，相机姿态，边界，用于合成的螺旋路径上的相机位姿，保留视图的索引
        hwf = poses[0,:3,-1]  # 从姿态矩阵中得到图像的hwf信息（前三行最后一列）
        poses = poses[:,:3,:4]  # 得到所有相机姿态矩阵3*4（3列xyz，一列t（平移量或者相机中心））
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):  # 如果i_test不是列表就给他转成列表形式
            i_test = [i_test]

        if args.llffhold > 0: # 如果这个参数大于0
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold] # 挑选样本生成测试集，i_test数组中的元素是images数组中每隔args.llffhold个位置的索引

        i_val = i_test # 将测试集样本索引赋予验证集
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)]) # 数据集中既不属于测试集也不属于验证集的样本索引，分配给训练集用

        print('DEFINING BOUNDS')
        if args.no_ndc:  # 如果这个参数存在
            near = np.ndarray.min(bds) * .9  # 近场景边界乘0.9得到近平面
            far = np.ndarray.max(bds) * 1.  # 远场景边界乘1作为远平面
            
        else: # 如果参数不存在，近远平面就直接写
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':  # 如果使用blender数据集
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip) # 加载blender数据集，得到所有样本图像，对应相机姿态矩阵，用于合成视图的螺旋路径上的相机姿态，图像hwf，所有样本的索引（划分好三个集后连续的索引）
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split  # 一个整集按三个集划分好后，属于他们的样本在所有样本中的索引.测试集的样本的索引存在i_train中
        # i_train 100,i_val 13,i_test 25
        near = 2. # 近远平面直接定
        far = 6.

        if args.white_bkgd: # 如果命令中有这个标志
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:]) # 将每个图像的RGB值乘上对应alpha值再加上对应的1-alpha 提取到用作图像
        else:
            images = images[...,:3]  # 直接用RGB信息作为图像

    elif args.dataset_type == 'LINEMOD': # 如果使用Linemod数据集
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip) # # 加载Linemod数据集，得到所有样本图像，对应相机姿态矩阵，用于合成视图的螺旋路径上的相机姿态，图像hwf，内参矩阵，所有样本的索引（划分好三个集后连续的索引），近远平面
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split # 三个集分到的样本索引

        if args.white_bkgd: # 如果命令中有这个参数
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:]) # # 将每个图像上像素的RGB值乘上对应alpha值（用RGB表示的带有透明度的像素值）再加上对应的1-alpha（RGB值与不透明度相加，得到与白色混合的效果），从而形成带有白色背景的效果。极端情况alpha=0也就是全透明，经过这一操作RGB为归一化后的1也就是原来的255，白色
        else:
            images = images[...,:3]  # 直接用RGB信息作为图像

    elif args.dataset_type == 'deepvoxels':  # 如果用deepvoxels数据集

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip) # 读取数据集得到，三个集的样本图像，样本对应相机姿态，测试时用于合成视图的相机姿态，hwf，所有样本的索引（划分好三个集后连续的索引）

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split # 分配样本索引

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1)) # 得到所有相机姿态矩阵中z轴上的向量长度的平均，最终得到在z轴朝向上的平均位置
        near = hemi_R-1. # 用这作为近平面
        far = hemi_R+1.  # 远平面

    else: # 四种数据集类型都不是
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]  # HW全转成整型后重新塞到hwf里

    if K is None: # 这个矩阵K是相机的内参矩阵，如果K没提供，这里就人工写一个：这里cx和cy的位置是用图像宽和高的一半近似，并且理想情况fx=fy这里统一用focal
        K = np.array([ # nerf里假设相机的内外参数是由数据集提供的，但要分合成数据集和真实数据集两种情况
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test: # 如果命令里有这个参数,或者说是这个参数设为了true
        render_poses = np.array(poses[i_test]) # 把属于测试集的样本的相机姿态拿出来给render_poses,用于合成视图，这样合成的视图就会有真实图参照比较，不然纯load到的render_poses就是一些生成的相机姿态，纯为了可视化的渲染，没有对照不分好坏

    # 存日志操作。下面这一小段完成的操作是在basedir/expname(log/blender_paper_lego)路径下创建args.txt以及config.txt文件，里面分别存的是本次训练时用到的所有arg属性和值，本次训练读取的那个配置文件（configs/lego.txt）中的所有配置内容
    basedir = args.basedir # 提取命令里面给出的路径
    expname = args.expname # 文件夹名称
    os.makedirs(os.path.join(basedir, expname), exist_ok=True) # 在指定路径上创建文件夹，就算原来存在也不会报错
    f = os.path.join(basedir, expname, 'args.txt') # f表示这个路径
    with open(f, 'w') as file: # 打开这个路径下的文件，主要想生成args.txt里面的内容
        for arg in sorted(vars(args)): # 遍历命令行参数对象 args 中的所有属性
            attr = getattr(args, arg) # 获取命令行参数对象args中属性名为arg的值
            file.write('{} = {}\n'.format(arg, attr)) # 在文件中写入属性名和属性值，格式为：arg=attr
    if args.config is not None: # 如果给出了配置文件
        f = os.path.join(basedir, expname, 'config.txt') # 生成要写入的配置文件的路径
        with open(f, 'w') as file: # 打开配置文件路径
            file.write(open(args.config, 'r').read()) # 以阅读方式打开命令行参数config指定的配置文件，读取内容，把内容写入这个生成的配置文件路径对应的文件下

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args) #把命令行参数输入后，得到训练要用的参数列表（字典形式），测试要用的参数列表，训练时的迭代次数，要优化的梯度变量们，优化器
    global_step = start # 学习率衰减时要用到的全局步数

    bds_dict = {
        'near' : near,
        'far' : far,
    }  # 边界字典
    render_kwargs_train.update(bds_dict) # 把near和far参数也写进训练要用的参数列表里
    render_kwargs_test.update(bds_dict) # 写进测试要用的参数列表里

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device) # 把用于视图合成的相机姿态转移到gpu上

    # Short circuit if only rendering out from trained model
    if args.render_only: # 为true表示，不优化、重新加载权重并渲染出路径render_poses。命令里有这个相当于直接做预测，不训练
        print('RENDER ONLY')
        with torch.no_grad(): # 以下不进行梯度计算
            if args.render_test: # 为true表示，渲染测试集而不是render_poses路径
                # render_test switches to test poses
                images = images[i_test] # 提取测试集的图像记录在images中，这个算真实值
            else:
                # Default is smoother render_poses path
                images = None # 因为没用测试集直接用生成的更顺滑的渲染路径，所以没有真实值图片做参考

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start)) # {:06d}一个零填充的六位整数，用于表示渲染开始的时间戳 start。这个是渲染出来的图片将存放的路径。‘test'表示内容是用测试集里面的搞出来的，’path‘表示内容是生成的更顺滑的渲染路径搞出来的
            os.makedirs(testsavedir, exist_ok=True) # 创建这个路径相关的目录
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor) # 返回所有相机姿态能得到的颜色图，和视差图
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)  # 先将颜色图都写回255的形式，然后将这个图像序列数据以视频的形式写入这个文件中

            return

    # Prepare raybatch tensor if batching random rays # 从一张图上提像素点和几张图像上的差别
    N_rand = args.N_rand # 每次梯度迭代随机选择的射线数量
    use_batching = not args.no_batching # args.no_batching为true表示一次只从一个图像中提取随机射线，否则这个随机射线可能来自很多图中，这边的这个变量反过来
    if use_batching: # 如果要从很多图片中选择出随机的射线，就做下面的操作，为更下面的操作做准备
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0)  # [N, ro+rd, H, W, 3] # 从不同的相机姿态矩阵中生成对应射线，最终所有的射线都在rays中
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3] # N是射线数量，这个变量表示里面不仅有射线还有图像信息
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only 因为上面生成的射线太多，这一步表示只提取部分射线用于训练（根据i_train中的索引）
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb) # 把上面得到的rays_rgb随机打乱

        print('done')
        i_batch = 0 # 用这个记录当前所在的批次

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 2000 + 1 #训练的结束次数200000 + 1被我改了
    print('Begin')
    print('TRAIN views are', i_train) # 打印出用于训练的样本索引
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):  # 训练epoch循环从start开始到N_iters-1，使用trange可以在循环中显示进度条，并实时更新进度。start可能是0，也可能加载了checkpoit文件，就不是1了，是在那个的训练基础上再训练
        time0 = time.time()  # 获取当前的时间戳

        # Sample random ray batch
        if use_batching: # 如果为true，就是要从很多图片中选择出随机的射线
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?] 上面已经得到了rays_rgb，里面是一些来自不同图片的射线和rgb信息。这里再按照批次i_batch，每次从里面找N_rand条赋予batch用于下面的操作。所以batch可以表示这一批次中要处理的射线信息和rgb
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2] # 从batch中提取得到这一批次的射线信息存在batch_rays里，这些射线对应的rgb信息存在target_s

            i_batch += N_rand  # 改变一下i_batch的数值，便于下一循环提取射线信息batch
            if i_batch >= rays_rgb.shape[0]: # 如果rays_rgb里每一个射线信息都被拿出来用过了，就把rays_rgb里面的信息再打乱，给下一个epoch用
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0]) # 生成一个这个形状的随机排列索引
                rays_rgb = rays_rgb[rand_idx]  # 因为索引是随机的，所以相当于完成了打乱操作
                i_batch = 0  # i_batch清零

        else:  # 这个情况是要从一个图片中选择出随机的射线来处理
            # Random from one image
            img_i = np.random.choice(i_train) # 随机选择训练集中的一个样本索引
            target = images[img_i] # 根据索引，提取对应的图像作为target[400,400,3]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]  # 提取这个样本索引对应的相机姿态矩阵

            if N_rand is not None: # 如果给出了N_rand
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)首先根据这一个相机姿态矩阵，得到这个视角下所有的射线信息（原点+方向）

                if i < args.precrop_iters:  # 如果当前的训练epcho:i，小于这个参数（这个参数表示需要需要使用中心裁剪的训练步数），就要做下面的操作生成图像一个中心区域的像素坐标
                    dH = int(H//2 * args.precrop_frac)  # 用于确定中心区域的高度范围（因为H//2得到是中心点的高度，又乘了后面这个相当于把高度往下了一点（中心点下面一点））
                    dW = int(W//2 * args.precrop_frac)  # 用于确定中心区域的宽度范围
                    coords = torch.stack(
                        torch.meshgrid(  # 得到中心区域内每个像素点坐标
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),  # 得到中心区域的行坐标。因为从H//2 - dH~H//2 + dH - 1要采样2*dH个点
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)   # 得到中心区域的列坐标
                        ), -1)
                    if i == start: # 如果刚开始第一个epoch训练，需要打印一下下面的信息
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else: # 超过给定范围的后面几个epoch，直接做下面的操作，生成图像中所有区域的像素坐标
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                # 生成的coord要么是中心区域的每一个像素点索引和坐标，要么是整个图像区域的像素点索引和坐标
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2) # 重塑形状，表示每个像素点的索引和坐标
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand) # 从coord中随机得到N_rand个索引值，（表示像素位置的索引）
                select_coords = coords[select_inds].long()  # (N_rand, 2) 将索引值对应的coord中的元素取出，得到我们选中的像素点坐标
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3) # 得到这些像素点对应的射线的原点信息
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3) # 得到这些像素点对应的射线的方向信息
                batch_rays = torch.stack([rays_o, rays_d], 0) # [2,N_rand,3]连接原点和方向信息，得到这一训练批次中所有射线的信息
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3) # 得到这些像素点的对应rgb颜色，存在target里面，用于后面的比较
        # 至此已经得到了训练要用的射线信息batch_rays和他们对应的颜色标签target_s
        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train) # 把这些射线送入函数，得到射线对应的颜色，视差，积累和其他额外的信息。相当于预测值（注意这里返回的是细网络的结果，或者是没有细网络存在时的粗网络结果）

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s) # 计算（细网络）预测的射线颜色和真实颜色之间的均方损失
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss) # 再根据得到的损失求评价标准psnr信噪比

        if 'rgb0' in extras:  # 如果上面射线经过处理后得到的额外信息中，有rgb0得到的信息。（也就是说用了粗细网络，上面的rgb是细网络输出，rgb0是粗网络输出）
            img_loss0 = img2mse(extras['rgb0'], target_s) # 计算粗网络预测rgb与真实值之间的均方差
            loss = loss + img_loss0 # 粗细网络分别求出的损失值相加，一起完成下面训练的迭代
            psnr0 = mse2psnr(img_loss0) # 这是由粗网络损失计算得到的信噪比psnr0

        loss.backward() # 计算梯度
        optimizer.step() # 根据梯度信息来更新模型的参数

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1 # 用于调整学习率的衰减率设置为0.1
        decay_steps = args.lrate_decay * 1000 # 表示衰减步数（表示在多少步之后进行学习率的衰减）
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))  # 计算新的学习率，将初始学习率args.lrate乘以衰减率的全局步骤数除以衰减步数的幂进行计算得到的
        for param_group in optimizer.param_groups: # 对优化器的参数组进行循环遍历
            param_group['lr'] = new_lrate # 将参数组的学习率设置为新的学习率
        ################################

        dt = time.time()-time0 # 得到上面这些操作进行一次所需要的时间
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0: # 每进行args.i_weights轮，就要做下面的操作,保存检查点文件，如果有检查点文件存在上面有读取操作
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(), # 返回模型的状态字典，即包含了模型的所有参数和对应的权重
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path) # 在这个路径下，保存上面这些信息
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:  # 每进行args.i_video轮，就要做下面的操作
            # Turn on testing mode
            with torch.no_grad(): # 下面不用梯度计算
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)  # 输入用于视图合成的相机姿态，得到对应颜色图和视差图
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))  # 构建保存视频文件的基本路径moviebase
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)  # 将生成的RGB序列图像保存为视频文件，在这个路径下
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)  # 将视差图归一化后与上面rgb做一样的操作保存为视频文件

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:  # 每进行args.i_testset轮，就要做下面的操作
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i)) # 路径
            os.makedirs(testsavedir, exist_ok=True) # 创建对应路径目录
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir) # 由测试集中某个姿态矩阵得到的渲染图片poses[i_test]，会被存放在这个路径下testsavedir
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}") # 在进度条中打印自定义文本
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor') # 设置默认的张量类型为 CUDA上的浮点型张量。这意味着在后续使用 torch.Tensor() 创建张量时，默认会创建一个存储在 GPU 上的浮点型张量

    train()
