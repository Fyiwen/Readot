import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from Dataset import CatsAndDogsDtaset

my_transforms=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)), # 大小变了，内容不会缺
    transforms.RandomCrop((224,224)), # 因为裁剪所以内容会缺
    transforms.ColorJitter(brughtness=0.5), # 将图像的亮度随机变化为原图亮度的（1-0.5）-（1+0.5）
    transforms.RandomRotation(degrees=45), # 随机进行图像旋转，随机在（-45，45）之间找个角度
    transforms.RandomHorizontalFlip(p=0.5), #根据概率来决定每次是否需要对图片进行水平翻转
    transforms.RandVerticalFlip(p=0.05),
    transforms.RandomGrayScale(p=0.2),
    transforms.Totensor(),
    transforms.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]) # 用均值和方差对像素值归一化
    
])
dataset=CatsAndDogsDataset(csv_file='cats_dogs.csv',root_dir='cats_dogs_resized',transform=my_transforms)

img_num=0
for _ in range(10):
    for img,label in dataset:
        save_image(img,'img'+str(img_num)+'.png')
        img_num+=1


# 以下关于另一个数据增强库
import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image

image=Image.open("images/elon.jpeg")
mask=Image.open("images/mask.jpeg")
mask2=Image.open("images/second_mask.jpeg")
transform=A.Compose(
    [
        A.Resize(width=1920,height=1080), # 调整图像到指定大小
        A.RandomCrop(width=1280,height=720), # 随机裁剪图像到指定大小
        A.Rotate(limit=40,p=0.9,border_mode=cv2.BORDER_CONSTANT),#以概率p对图像进行旋转，旋转角度限制为-40-40，对超出边界的区域进行常数填充
        A.HorizontalFlip(p=0.5), # 以一定概率进行水平翻转
        A.VerticalFlip(p=0.1), # 垂直翻转
        A.RGBShift(r_shift_limit=25,g_shift_limit=25,b_shift_limit=25,p=0.9), # 以一定概率对rgb通道进行偏移改变，这边三个通道的改变范围都是-25-25
        A.OneOf([
            A.Blur(blur_limit=3,p=0.5), # 以一定概率对图像进行模糊处理，模糊程度限制为3
            A.ColorJitter(p=0.5), #颜色抖动操作会随机地改变图像的亮度、对比度、饱和度和色相等属性
        ],p=1.0) # 以概率1做这个操作，然后是选里面的一个执行
    ]
)
images_list=[image]
image=np.array(image)
mask=np.array(mask)
mask2=np.array(mask2)
for i in range (15):
    augmentations=transform(image=image,masks=[mask,mask2])# 这三张图片全都经过了transform的处理
    augmented_img=augmentations["image"] # 提取image这张图片对应的结果
    augmented_masks=augmentations["masks"] 
    images_list.append(augmented_img) # 将结果图片放到列表中，方便后面显示出来做对比
    images_list.append(augmented_masks[0])
    images_list.append(augmented_masks[1])
plot_examples(images_list) # 最终会显示图片，改变前和改变后的

#对于目标检测，可能会用到的数据增强
import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image

image=cv2.imread("images/cat.jpg")
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # 转图片颜色空间的操作
bboxes=[[13,170,224,410]] # 这个边界框是按照Pascal_voc的写法左下右上，COCo、YOLO可能表示不一样。这里是手动假设存在一个这样的检测框

transform=A.Compose(
    [
        A.Resize(width=1920,height=1080),
        A.RandomCrop(width=1280,height=720),
        A.Rotate(limit=40,p=0.9,border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25,g_shift_limit=25,b_shift_limit=25,p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3,p=0.5),
            A.ColorJitter(p=0.5),
        ],p=1.0) # 以概率1做这个操作，选里面的一个执行
    ],bbox_params=A.BboxParams(format="pascal_voc",min_area=2048,min_visibility=0.3,label_fields=[]) # 边界框的数据增强就是用这最后个。指定边界框的坐标格式为Pascal VOC格式。指定边界框的最小面积阈值为2048。较小的边界框面积将被过滤掉，不参与数据增强操作。指定边界框的最小可见度为0.3。可见度表示边界框在图像中的可见程度，取值范围为0到1。可见度低于0.3的边界框会被过滤掉，不参与数据增强操作。指定边界框的标签字段为空列表。在一些数据集中，可能会给每个边界框赋予一个或多个标签，这里可以指定包含边界框标签的字段名称。
)
images_list=[image]
saved_bboxes=[bboxes[0]]
for i in range (15):
    augmentations=transform(image=image,bboxes=bboxes) # 对边界框和图像框都进行变换处理
    augmented_img=augmentations["image"]
    if len(augmentations["bboxes"])==0:
        continue
    images_list.append(augmented_img)
    saved_bboxes.append(augmentations["bboxes"][0])
plot_examples(images_list,saved_bboxes)

#
import torch
import numpy as np
import cv2
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

class ImageFolder(nn.Module):
    def __init__(self,root_dir,transform=None):
        super(ImageFolder,self).__init__()
        self.data=[]
        self.root_dir=root_dir
        self.transform=transform
        self.class_names=os.listdir(root_dir)

        for index,name in enumerate(self.class_names):
            files=os.listdir(os.path.join(root_dir,name))
            self.data+=list(zip(files,[index]*len(files)))

    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img_file,label=self.data[index]
        root_and_dir=os.path.join(self.root_dir,self.class_names[label])
        image=np.array(Image.open(os.path.join(root_and_dir,img_file)))

        if self.transform is not None:
            augmentations=self.transform(image=image) # 对image进行数据增强
            image=augmentations["image"]
        return image,label

transform=A.Compose(
    [
        A.Resize(width=1920,height=1080),
        A.RandomCrop(width=1280,height=720),
        A.Rotate(limit=40,p=0.9,border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25,g_shift_limit=25,b_shift_limit=25,p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3,p=0.5),
            A.ColorJitter(p=0.5),
        ],p=1.0),# 以概率1做这个操作，选里面的一个执行
        A.Normalize(
            mean=[0,0,0],
            std=[1,1,1],
            max_pixel_value=255,
        ), # 规范化公式为img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        A.ToTensorV2(),# 也是转张量操作
    ]
)
dataset=ImageFolder(root_dir="cat_dogs",transform=transform)