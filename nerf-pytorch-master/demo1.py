import numpy as np
import matplotlib.pyplot as plt


def normalize(x):
    return x / np.linalg.norm(x)  # x除以他的范数值做归一化


def viewmatrix(z, up, pos):  # 输入是相机的Z轴朝向、up轴的朝向(即相机平面朝上的方向Y)、以及相机中心，输出的是c2w矩阵，由旋转矩阵R和平移矩阵T组成
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))  # 用z轴和y轴的叉乘得到关于x轴方向的信息，因为三轴互相垂直
    vec1 = normalize(np.cross(vec2, vec0))  # 也是用叉乘得到y轴方向信息，因为传入的up(Y)轴是通过一些计算得到的，不一定和Z轴垂直
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m  # 这是c2w矩阵，描述相机的位置和朝向


def poses_avg(poses):  # 输入是多个相机的位姿，输出多个相机的平均位姿（包括位置和朝向）。这里poses形状为N*3*5

    hwf = poses[0, :3, -1:]  # 取第一个相机位姿的前三行和最后一列，也就是得到形状为3*1是图像的长宽和焦距。

    center = poses[:, :3, 3].mean(0)  # 对多个相机的中心进行求均值得到center
    vec2 = normalize(poses[:, :3, 2].sum(0))  # 对所有相机的Z轴求平均得到vec2向量（方向向量相加其实等效于平均方向向量）。
    up = poses[:, :3, 1].sum(0)  # 对所有的相机的Y轴求平均得到up向量
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf],
                         1)  # 将得到的vec2, up和center输入到viewmatrix()函数就可以得到平均的相机位姿4*4,然后再连接3*1得到4*5的新c2w矩阵

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots,N):  # 为了生成一段螺旋式的相机轨迹用于新视角的合成（具体的：相机绕着一个轴旋转，其中相机始终注视着一个焦点，相机的up轴保持不变）。zdelta: 相机在焦点和相机位置之间的初始距离。zrate: 控制相机在旋转过程中沿着 Z 轴移动的速度。rots: 控制相机绕着焦点旋转的圈数。
    render_poses = []  # 用于储存得到的相机位姿
    rads = np.array(list(rads) + [1.])  # rads转换为一个带有额外元素1.0的NumPy数组，多了1可能是为了长度和其他数组对齐。rads应该是相机绕焦点旋转的半径值
    hwf = c2w[:, 4:5]  # 将c2w的第五列提取出来作为hwf数组

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:  # 每一迭代生成一个新的相机位置theta，这个的取值来自从0-2pi*rot之间均匀分布的值，除了最后一个
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate),
                                          1.]) * rads)  # 指当前迭代的相机中心在世界坐标系的位置，这一步又实现了旋转，旋转时沿z轴移动，而且旋转时的半径也由rads提供可能发生改变
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal,
                                                        1.])))  # np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])得到焦点在世界坐标系的位置，然后归一化，相当于单位化得到的z是相机z轴在世界坐标系的朝向
        render_poses.append(
            np.concatenate([viewmatrix(z, up, c), hwf], 1))  # 用viewmatrix(z, up, c)构造当前相机的矩阵，然后再和hwf拼接得到4*5矩阵
    return render_poses

def recenter_poses(poses):  # 输入N个相机位姿，返回N个相机位姿。为的是中心化每个相机位姿

    poses_ = poses + 0  # 相当于创建了poses数组的一个副本赋值给poses_
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])  # 将一个列表重塑成1*4矩阵
    c2w = poses_avg(poses)  # 得到多个输入相机的平均位姿c2w是4*5的，前三列对应xyz，第四列对应相机中心，第五列对应hwf
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)  # 3*8
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]),
                     [poses.shape[0], 1, 1])  # 重塑后的三维bottom再三个维度上分别重复poses第一个维度的大小N次,1次,1次，形状为N*1*4
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses  # 用这个平均位姿c2w的逆矩阵左乘到输入的相机位姿上就完成了归一化。因为同一个旋转平移变换矩阵左乘所有的相机位姿是对所有的相机位姿做一个全局的旋转平移变换
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):  # 这个函数可以用于"球面化"相机分布并返回一个环绕的相机轨迹用于新视角合成，以得到相机围绕着一个物体拍摄的360度场景
    # 前半部分是在将输入的相机参数进行归一化，后半部分是生成一段相机轨迹用于合成新视角
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])],
                                         1)  # 用于将输入的3x4矩阵p扩展为4x4矩阵，取单位矩阵最后一行，重塑后重复，再和p做连接

    rays_d = poses[:, :3, 2:3]  # 提取出所有相机的方向向量（也就是中心射线方向），形状（n，3，1）
    rays_o = poses[:, :3, 3:4]  # 提取出所有相机的中心坐标（也就是中心射线原点），形状（n，3，1）

    def min_line_dist(rays_o, rays_d):  # 为找到离所有相机中心射线距离之和最短的点（可以先简单理解成场景的中心位置） #################
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])  # 一个投影矩阵
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)  # 将得到的最优点位置设为场景的中心（即新的相机轨迹的中心）
    # 下面将得到的场景中心位置移到世界坐标系的原点，同时将所有相机z轴的平均方向转到和世界坐标系的z轴相同
    # 就是以场景中心为原点构造一个新的坐标系，实现新坐标系和世界坐标系之间的转换
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)  # 计算所有相机中心坐标与场景中心位置的差的平均值，并将结果赋值给变量 up

    vec0 = normalize(up)  # 计算出新坐标系的三个单位向量
    vec1 = normalize(np.cross([.1, .2, .3], vec0))  # 只要选一个与vec0不共线的向量，就可以求到与vec0垂直的向量，[1,2,3]的选择比较常见
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center  # 将场景中心位置设为新的相机坐标系的原点
    c2w = np.stack([vec1, vec2, vec0, pos], 1)  # 形成新的c2w矩阵

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])  # 现在相机姿态已经从世界坐标系下转换成这个新相机坐标系下的表示

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))  # 计算在新的相机坐标系下，所有相机位置离场景中心的平均距离
    # 下面将相机的位置缩放到单位圆内
    sc = 1. / rad  # 缩放因子
    poses_reset[:, :3, 3] *= sc  # 相机位置缩放
    bds *= sc  # 场景边界缩放
    rad *= sc  # 半径缩放

    centroid = np.mean(poses_reset[:, :3, 3], 0)  # 计算新的相机轨迹的中心位置
    zh = centroid[2]  # 拿出中心位置的z坐标赋值
    radcircle = np.sqrt(rad ** 2 - zh ** 2)  #
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):  # 等间距数组表示的角度值，用于确定各个相机在圆周上的位置

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])  # 得到相机位置在xy平面上的坐标，z轴坐标是固定的zh
        up = np.array([0, 0, -1.])  # 表示了相机的朝向，这里朝-z轴方向

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin  # 相当于平移量，也是相机中心位置
        p = np.stack([vec0, vec1, vec2, pos], 1)  # 形成相机姿态矩阵

        new_poses.append(p)  # 得到新的相机轨道上，所有相机的位置

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)],
        -1)  # 一开始输入的相机姿态，只是换成了新相机坐标系的表示

    return poses_reset, new_poses, bds


def load_llff_data(recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    '''poses=np.array([
    [[0.33,1,1.9,3],
    [0.4,0.99,0.22,0.43],
    [2.1,0.222,1.43,1.3],
    [0,0,0,1]],[
    [0.88,0.932,1.4,1.7],
    [1.8,0.44,2.3,0.9],
    [3.5,2.11,0.53,2.4],
    [0,0,0,1]]])'''
    poses=np.array([
        [[
                    -0.9999021887779236,
                    0.004192245192825794,
                    -0.013345719315111637,
                    -0.05379832163453102
                ],
                [
                    -0.013988681137561798,
                    -0.2996590733528137,
                    0.95394366979599,
                    3.845470428466797
                ],
                [
                    -4.656612873077393e-10,
                    0.9540371894836426,
                    0.29968830943107605,
                    1.2080823183059692
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]],[
                [
                    -0.9305422306060791,
                    0.11707554012537003,
                    -0.34696459770202637,
                    -1.398659110069275
                ],
                [
                    -0.3661845624446869,
                    -0.29751041531562805,
                    0.8817007541656494,
                    3.5542497634887695
                ],
                [
                    7.450580596923828e-09,
                    0.9475130438804626,
                    0.3197172284126282,
                    1.2888214588165283
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ],[
                [
                    0.4429636299610138,
                    0.31377720832824707,
                    -0.8398374915122986,
                    -3.385493516921997
                ],
                [
                    -0.8965396881103516,
                    0.1550314873456955,
                    -0.41494810581207275,
                    -1.6727094650268555
                ],
                [
                    0.0,
                    0.936754584312439,
                    0.3499869406223297,
                    1.4108426570892334
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ],[
                [
                    0.7956318259239197,
                    0.5209260582923889,
                    -0.3092023432254791,
                    -1.2464345693588257
                ],
                [
                    -0.6057805418968201,
                    0.6841840147972107,
                    -0.40610620379447937,
                    -1.6370664834976196
                ],
                [
                    -1.4901161193847656e-08,
                    0.5104197859764099,
                    0.859925389289856,
                    3.4664700031280518
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
    ])
    bds=[2,6]

    if recenter:  # 如果为true，就中心化每个相机的位姿
        poses = recenter_poses(poses)

    if spherify:  # 如果为true，就对姿态矩阵进行球面化的操作
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:  # 否则

        c2w = poses_avg(poses)  # 计算所有相机姿态的平均位姿

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))  # 对所有相机姿态矩阵中有关y轴方向的向量取和，相当于求平均，在归一化得到up轴

        focal = 3 # 用平均深度作为焦距

        # Get radii for spiral path
        shrink_factor = .8  # 缩小因子，用于控制螺旋路径的大小
        zdelta = 1,8 * .2  # 用近焦点深度求到的值*0.2，作为沿着Z轴的位移量，用于控制相机在螺旋路径上的垂直移动。
        tt = poses[:, :3, 3]   # 提取所有相机姿态矩阵的平移部分
        rads = np.percentile(np.abs(tt), 90,0)  # 计算这个新数组在列方向上的百分数为90的结果。即找到一个数能使tt中90%值包含在这个数范围内。计算结果每列对应一个，这个结果的计算依靠loc = 1 + (n - 1) * p ，其中n为排序后数的个数，p为百分位，num是结果 = a[loc整数部分 - 1] + (a[loc整数部分] - a[loc整数部分 - 1]) * loc小数部分，a就是排序后的新数组
        c2w_path = c2w  # 相机的平均位姿
        N_views = 120  # 螺旋路径中观察点数量
        N_rots = 2  # 螺旋路径中旋转次数
        if path_zflat:  # 一旦为true
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -1.8 * .1  # 设置一个沿着Z轴的偏移量，用于将相机沿着Z轴向下移动。
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]  # 将原来的相机姿态矩阵在Z轴方向上进行偏移。
            rads[2] = 0.  # 将螺旋路径中的Z轴方向上的半径设为0，使相机沿着Z轴不发生旋转。
            N_rots = 1  # 旋转次数改
            N_views /= 2  # 观察点个数也改

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots,
                                          N=N_views)  # 产生旋转路径上的相机姿态矩阵们

    render_poses = np.array(render_poses).astype(np.float32)

    return render_poses


ax = plt.figure().add_subplot(projection='3d') # 创建一个三维坐标系，并将其存储在变量 ax 中


p0 = np.array([100,20,100,1]) # 这个p0作为初始点
ax.scatter(p0[0], p0[1], p0[2], c = 'r', label='init points') # p0中三个表示要绘制散点的x、y和z坐标，相当于在三维坐标系上绘制定点p0，红色，标签为...
ax.legend() # 把之前绘制图形时指定的标签添加到图例中
ax.set_xlabel('X') # 设置三维坐标系中的x轴标签为X
ax.set_ylabel('Y')
ax.set_zlabel('Z')

step=1
new_row = np.array([0, 0, 0, 1])
b = load_llff_data(recenter=True, bd_factor=.75, spherify=False, path_zflat=False)
for i in range(0, 120, step):
    s1=b[i]
    s2=s1[:,:4]
    s = np.vstack((s2, new_row))
    p = s@ p0 # 用这个和下面那个都可以，这个就是直接拿一个真实点p0，乘上每一个变换的矩阵c2w，可以看到新得到的点之间的变化，轨迹和下面是类似的
    #p=s[:,-2] # 用这个的话就是直接看每个c2w矩阵的最后一列，也就时平移对应的内容，里面的数既表示相机坐标原点相对于世界原点的偏移量，也表示在世界坐标系下这个相机原点的坐标
    ax.scatter(p[0], p[1], p[2]) # 把产生的新点p加到三维坐标系里
    plt.show(block=False) # 使得显示图形后不会阻塞程序的执行，而是继续执行后面的代码
    plt.pause(1) # 使绘图过程中暂停一段时间，使得观察者能够有机会查看图像的变化
    print(p)


# Make legend, set axes limits and labels
ax.legend()
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
# ax.view_init(elev=20., azim=-35, roll=0)

plt.show()