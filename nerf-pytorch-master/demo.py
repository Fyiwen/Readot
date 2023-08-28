import numpy as np
import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection='3d') # 创建一个三维坐标系，并将其存储在变量 ax 中

trans_t =  lambda t: np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
])

rot_phi = lambda phi: np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
])

rot_theta = lambda th: np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
])

p0 = np.array([100,20,100, 1]) # 这个p0作为初始点
ax.scatter(p0[0], p0[1], p0[2], c = 'r', label='init points') # p0中三个表示要绘制散点的x、y和z坐标，相当于在三维坐标系上绘制定点p0，红色，标签为...
ax.legend() # 把之前绘制图形时指定的标签添加到图例中
ax.set_xlabel('X') # 设置三维坐标系中的x轴标签为X
ax.set_ylabel('Y')
ax.set_zlabel('Z')

step = 9
allp = [p0]

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)  # [4,4]
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

#pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]]

for theta in range(-180, 180, step):
    s = pose_spherical(theta,-30.0,4.0)
    p = s@ p0 # 用这个和下面那个都可以，这个就是直接拿一个真实点p0，乘上每一个变换的矩阵c2w，可以看到新得到的点之间的变化，轨迹和下面是类似的
    #p=s[:,-1] # 用这个的话就是直接看每个c2w矩阵的最后一列，也就时平移对应的内容，里面的数既表示相机坐标原点相对于世界原点的偏移量，也表示在世界坐标系下这个相机原点的坐标
    ax.scatter(p[0], p[1], p[2]) # 把产生的新点p加到三维坐标系里
    plt.show(block=False) # 使得显示图形后不会阻塞程序的执行，而是继续执行后面的代码
    plt.pause(1) # 使绘图过程中暂停一段时间，使得观察者能够有机会查看图像的变化
    print(p)
    allp.append(p)

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