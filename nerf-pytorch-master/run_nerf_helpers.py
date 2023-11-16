import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8) # 将输入的x转换成取值范围在0到255之间的无符号8位整数（uint8）类型。np.clip(x, 0, 1) 将数组 x 中的元素限制在区间 [0, 1] 内。小于0的值会被截断为0，大于1的值会被截断为1。unit8是0-255之间的整数


# Positional encoding (section 5.1)
class Embedder:  # 这个类有了生成嵌入器需要的参数作为输入后，用于生成嵌入器对象
    def __init__(self, **kwargs):  # **kwargs是调用这个类时会给出的输入参数
        self.kwargs = kwargs  # 现在self.kwargs这里的就是那些给的嵌入器参数
        self.create_embedding_fn()  # 定义了一个函数在下面有具体
        
    def create_embedding_fn(self):  # 创建嵌入函数
        embed_fns = [] # 用于存储嵌入函数
        d = self.kwargs['input_dims'] # d从输入中提取得到输入数据的维度
        out_dim = 0 # 输出维度
        if self.kwargs['include_input']: # 如果为true，就把恒等函数x添加到embed_fns列表中，并将输入数据的维度d加到out_dim中，表示在嵌入处理中保留原始输入数据
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2'] # 提取参数L-1
        N_freqs = self.kwargs['num_freqs'] # L
        
        if self.kwargs['log_sampling']: # 根据这个参数的值决定用哪个方式生成频率分段
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs) # 使用对数均匀采样方式生成频率分段。这里相当于完成[2^0,2^1,...,2^(L-1)]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs) # 指数均匀采样方式生成频率分段
            
        for freq in freq_bands:  # 遍历频率函数eg：[2^0,2^1,...,2^(L-1)]
            for p_fn in self.kwargs['periodic_fns']: # 遍历周期函数sin和cos
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq)) # 将一个完整的位置编码函数中每一部分的嵌入函数p_fn(x * freq)添加到embed_fns这个列表中。pi咋没了？？？理论上用Transformer那个位置编码是要有pi的
                out_dim += d # 每做一次嵌入嵌入就增加d个输出维度
                    
        self.embed_fns = embed_fns #可以完成位置编码的函数
        self.out_dim = out_dim
        
    def embed(self, inputs): # 这个函数在得到对应输入参数inputs后，会将输入运用到每一个嵌入函数fn中，再将结果拼接
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)  # 若self.embed_fns中有n个嵌入函数，那么最终的输出结果是一个形状为 (inputs.shape[0], self.out_dim * n)的张量，其中inputs.shape[0]是输入数据的样本数量，self.out_dim 是每个嵌入函数的输出维度


def get_embedder(multires, i=0):
    if i == -1:  # 如果i等于-1，就是不进行位置编码，则直接返回下面，否则执行更下面的代码
        return nn.Identity(), 3 # nn.Identity() 是一个恒等映射层，不对输入进行任何处理，因此在这种情况下，嵌入器被设定为恒等映射层，并且输出维度为3
    # 否则需要生成嵌入器对象
    embed_kwargs = { # 这里先生成嵌入器对象所需的一些参数
                'include_input' : True,
                'input_dims' : 3, # 输入位置编码前的维度
                'max_freq_log2' : multires-1,
                'num_freqs' : multires, # 公式中的L
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs) # 用上面给的参数生成嵌入器对象
    embed = lambda x, eo=embedder_obj : eo.embed(x) # 这个函数可以实现接收了x和嵌入器对象eo后，调用方法eo.embed(x)进行嵌入操作
    return embed, embedder_obj.out_dim # 返回嵌入函数和嵌入器对象的输出维度


# Model
class NeRF(nn.Module): # nerf模型
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False): # 构建网络的参数
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D # 网络层的深度。D个线性层，当然整个网络还有其他线性层，这D个只是可以一起处理
        self.W = W # 网络中一开始的输出通道数，反正就是通道数后面几层也用
        self.input_ch = input_ch # 输入中点x的特征通道数
        self.input_ch_views = input_ch_views # 输入中视角d的特征通道数
        self.skips = skips # 里面记录了一个层的索引，这个层需要特殊处理
        self.use_viewdirs = use_viewdirs # 决定了是否使用视角信息
        
        self.pts_linears = nn.ModuleList( # 包含了D=8个线性层。第一个线性层输入大小为 input_ch，输出大小为 W=256。其余的线性层根据 skips列表决定输入大小：如果当前层索引不在 skips中，则输入大小为 W，输出大小也为 W；如果当前层索引在 skips中，则输入大小为 W + input_ch，输出大小为W
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])  # 包含1个线性层。该线性层的输入大小为 input_ch_views + W，输出大小为128

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs: # 如果这个为真，多定义了三个线性层，不然就定义一个
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1) # 输出体积密度
            self.rgb_linear = nn.Linear(W//2, 3) # 输出颜色
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1) # 将输入x按照最后一个维度进行分割，分出表示点的特征和视角的特征。第一个子张量input_pts的大小为self.input_ch，第二个子张量input_views的大小为self.input_ch_views
        h = input_pts  # 表示输入中关于点的特征的部分
        for i, l in enumerate(self.pts_linears): # 遍历这里的D个线性层，把输入h塞到每一层中，结果再进relu层后再送到下一个线性层。其中有一个层（索引在self.skips中），需要将结果和input_pts进行拼接后再送入下一层
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs: # 如果为真，h还要送入下面的线性层中，进行处理
            alpha = self.alpha_linear(h) # 输出体积密度alpha
            feature = self.feature_linear(h) # 这一层既要一个正常输出256->256，还要与input_views连接，作为这一层真正的输出
            h = torch.cat([feature, input_views], -1) # w+input_ch_views
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h) # input_ch_views + W->128 # 这个层在论文图中没有画出来
                h = F.relu(h)

            rgb = self.rgb_linear(h) # 128->3得到输出颜色
            outputs = torch.cat([rgb, alpha], -1)  # 将颜色和体积密度对应写在一起
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights): # 实现了从Keras加载权重
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False" # 这个函数只在self.use_viewdirs为true时使用
        
        # Load pts_linears
        for i in range(self.D): # 对于self.pts_linears里面的D个线性层
            idx_pts_linears = 2 * i # 因为下面两行一次性要从weights里面拿走两个，所以这里索引从0开始，两个一弄
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))  #将weights中对应索引的权重转置之后转换为张量并赋值给相应的线性层的weight.data和bias.data
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D # 因为上面D个线性层，每个线性层w和b，用掉了2D个索引了，所以现在索引从2D开始
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2 # 因为上面用掉2D+2个了
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4 # 因为上面用掉2D+4个了
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6 # 因为上面用掉2D+6个了
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w): # 给定了一个相机姿态，和hwk，得到对应的射线信息，原点+方向们
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'生成一个网格坐标系（像素格的图像），其中i和j分别表示图像的行和列坐标。torch.linspace(0, W-1, W) 和 torch.linspace(0, H-1, H) 用于生成坐标点的范围。0开始，w-1结束长度为w等间距
    i = i.t() # 双双转置[400,400]
    j = j.t() # 转置后ij形成的坐标就符合像素坐标系了[400,400]
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)  # 计算每个像素对应的射线方向。用相机内参矩阵把像素坐标转换成相机坐标系下的射线方向。xyz方向[400,400,3]
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]把相机坐标系下的射线方向转换成世界坐标系下
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape) # 得到所有射线的原点，一个视角一个，世界坐标系下[400,400,3]
    return rays_o, rays_d # 得到这个c2w视角下，一个原点+好多方向


def get_rays_np(H, W, K, c2w): # 用于构造以相机中心为起始点，经过相机中心和图像像素点的射线，这里采取先在相机坐标系下用两点构建射线，再用c2m矩阵转换到世界坐标系的方法
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1) # 因为相机中心的坐标为（0，0，0），像素点3D坐标中x、y可以用2D图像坐标（i，j），减去光心坐标（cx，cy），然后z坐标其实就是焦距。因此可以得到射线的方向向量为（i-cx,j-cy,f）-(0,0,0),还可以把这个向量除以f归一化z坐标。但是这句代码里yz方向都多乘了一个负号，因为colmap里面相机坐标系的up/y朝下，相机光心朝向z正方向，而nerf相机坐标系里up/y朝上，相机光心朝z负方向，所以这两个相机坐标系之间还要切换一下
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):  # 把射线表示从相机坐标系下，转换成ndc坐标系下，用论文里推导出来的公式
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]  # (-n-oz)/dz
    rays_o = rays_o + t[...,None] * rays_d  # 将原来的相机光心o移到了近裁剪平面,所以得到了新原点o
    
    # Projection开始求投影矩阵
    # 射线原点o在三个方向上的分量
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]  # [-f/(w/2)]*(ox/oz)
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]  # [-f/(H/2)]*(oy/oz)
    o2 = 1. + 2. * near / rays_o[...,2]  # (1+2n/oz)
    # 射线方向d在三个方向上的分量
    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2]) # [-f/(w/2)][(dx-dz)/(ox-oz)]
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2]) # # [-f/(H/2)][(dy-dz)/(oy-oz)]
    d2 = -2. * near / rays_o[...,2] # (-2n/oz)
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d  # 得到转换后的射线信息


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False): # 输入新的的中间的采样点的深度值，每个原来的采样点的权重，还需要再采样的点的个数，因为det=(perturb==0.)所以不扰动采样点他就是true，以及pytest
    # bins[射线个数，每根射线上粗采样点数-1](因为是原来采样点的中心点所以少一个很正常),weights[射线个数，没跟射线上粗采样点数-2]？？？？？？？？？？？少两个（抛弃了最远和最近的两个值，下面cdf中会再补一个变成63和bins里面的63也是对应的）
    # Get pdf
    weights = weights + 1e-5 # prevent nans 给每个权重值加了一个很小的数【256，62】
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # 【256，62】把权重值做了归一化，这样每个值可以看成是一个概率（如果这个权重乘颜色的话，也就是说一个射线上所有采样点颜色乘上对应的概率，得到整个射线的颜色）
    cdf = torch.cumsum(pdf, -1)  # 【256，62】这样可以得到每个区间内（两个采样点之间）的累积概率
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  #【256，63】 (batch, len(bins))相当于在cdf左侧添加了一个0，使得cdf起始值为0（为了处理一种情况：这个采样点的概率小于最小的那个权重，就当作他的权重为0）

    # Take uniform samples
    if det:  # 如果为真就进行确定性采样，均匀地在[0, 1]的范围内生成N_samples个等间距的采样点，再扩展形状
        u = torch.linspace(0., 1., steps=N_samples) #
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])  # 把u扩展成的形状，是去掉cdf的最后一维，再加上一维，大小为N_samples(batch,N_samples )
    else:  # 否则使用随机数生成器在[0, 1]的范围内生成 N_samples个随机采样点，并且形状和上面一样
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]) #【256，128】

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest: # 如果有pytest，就要重新生成上面的采样点u，下面就算随机生成也是用的固定随机数
        np.random.seed(0)  # 定义了随机种子0
        new_shape = list(cdf.shape[:-1]) + [N_samples] # 定义了形状
        if det:
            u = np.linspace(0., 1., N_samples) # 这里还是确定性采样
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape) # 这里是固定随机数的采样
        u = torch.Tensor(u)
    # 反正得到的u是0-1之间的点，cdf是一个有序张量，里面每一个元素表示累积到当前的概率，越来越大
    # 接下来开始Invert CDF，指根据累积分布函数（CDF）的值，找到对应的随机变量的取值
    u = u.contiguous()  # 将u张量转换为连续的形式，以保证他在在内存中是连续的
    inds = torch.searchsorted(cdf, u, right=True)  # 在cdf张量中查找u张量对应的上限区间的索引。待搜索的有序张量（cdf）和要搜索的值（u）。inds中每个元素表示对应u在有序cdf中的插入点位置。right=True 表示使用右侧查找的方式。也就是说，inds中的每个元素都表示对应的u在CDF中找到的最靠右的位置。也就是说，对于每个u，它所在的区间的上限索引就是inds中对应位置的值
    below = torch.max(torch.zeros_like(inds-1), inds-1)  # 将 inds-1 张量中小于零的值替换为零。也就是得到下限索引张量。
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)  # 将inds张量中大于最大索引的值替换为最大索引。也就是上限索引张量。因为cdf.shape[-1]-1算出来的就是cdf中最大索引的值
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)连接对应的上下限索引张量，也就是说每个对应位置包含每个采样点所在的区间范围的下限索引和上限索引

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] # 设定了一个形状
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) # 根据inds_g中给定的索引，从指定的张量cdf中收集对应位置的值（上限和下限对应的概率）
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) # 根据inds_g中给定的索引，从指定的张量bin中收集对应位置的值（上限和下限对应的深度值）

    denom = (cdf_g[...,1]-cdf_g[...,0]) # cdf_g[...,1]表示CDF中每个值的上限，而cdf_g[...,0]表示CDF中每个值的下限。通过计算它们的差分，可以得到每个区间的长度
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom) # 将denom中小于1e-5的值替换为1，而保持其他的值不变。主要是为了下面的除法操作
    t = (u-cdf_g[...,0])/denom # t的值可以理解为采样点相对于所在区间的位置，t=0 表示采样点恰好位于区间的下限，而 t=1 表示采样点恰好位于区间的上限
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0]) # 使用线性插值的方式，根据t值在bins_g的一个个区间上进行取样，得到最终采样结果samples。bins_g[...,0]里面是每组深度值下限，bins_g[...,1]-bins_g[...,0]得到关于深度值的上限和下限的距离

    return samples # 得到采样结果，用深度值表示的
