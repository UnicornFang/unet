"""
  测试网络代码（生成预测图）

一、 引入相关库文件
二、 初始化网络相关参数
三、 对图片进行处理变换
四、 应用已训练好的模型分割图片
"""

"""================================= 一、 引入相关库文件 ======================================"""
from torchvision.transforms import transforms  # 引用库文件
from skimage import io
from matplotlib import pyplot as plt
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os, argparse, math, sys
import cv2
from torch.utils.data import DataLoader
import torch

' 引入自己的网络 '
# from Model.Swin_Unet_Quartet import SwinUnet_DAM
# from D_Model.Unet import Unet
from D_Model.Unet import Unet

"""*****************************************************************************************"""


"""================================ 二、 初始化网络相关参数 ================================="""
parser = argparse.ArgumentParser()  # 建立解析对象

' 加载模型 '
parser.add_argument('--model', type=str, default=Unet(), help='所使用的网络模型:{Unet(3, 1); SwinUnet(img_size=224, num_classes=1)}')

' 设置其他参数 '
parser.add_argument('--image_path', type=str, default=r"D:\Image_data_SR\0_Medical_images\0-Official-data\7_Skin_cancer_dataset\Skin_Cancer_dataset\big_data\test\Img", help='原图输入路径')
parser.add_argument('--out_path', type=str, default=r"E_Out_image/", help='预测图输出路径')
parser.add_argument('--weight_path', type=str, default=r"B_Weight/CiT_S/weights_25_0.899.pth", help='权重加载路径路径')
parser.add_argument('--image_type', type=str, default=r"jpg", help='原图输入类型；{png; jpg; bmp}')
parser.add_argument('--image_high', type=int, default=224, help='输出图片高度')
parser.add_argument('--image_width', type=int, default=224, help='输出图片宽度')

test_config = parser.parse_args()
"""******************************************************************************************"""

"""=============================== 三、 对图片进行处理变换 ===================================="""
' 对（inputs）输入图片的数据进行操作 '
x_transforms = transforms.Compose([
    transforms.ToTensor(),      # 将数据转为tensor类型，方便pytorch进行自动求导，优化之类的操作
    # transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 数据归一化，两个参数，一个为均值，一个为方差，均设置为0.5，每个参数里三个0.5表示有三个通道
    # transforms.Normalize([0.5], [0.5])  # 单通道
])
"""******************************************************************************************"""

"""========================== 四、 应用已训练好的模型分割图片 ================================="""

path = test_config.image_path
Filelist = os.listdir(path)  # 列举当前文件夹所有文件

def test_image():
    model = test_config.model
    ckpt = torch.load(test_config.weight_path, map_location='cpu')


    model.load_state_dict(ckpt, strict=False)
    model.eval()
    for pictureName in Filelist:
        ' 判断照片格式是否正确 '
        if pictureName[-3:] != test_config.image_type:
            continue
        ' 获取图片路径 '
        image_path = path + "/" + pictureName
        img_x = Image.open(image_path)
        # img_x = img_x.convert('L')    # 将图片转换为灰度图像
        ' resize图片尺寸 '
        img_x = img_x.resize((test_config.image_high, test_config.image_width))
        img_x = x_transforms(img_x)
        img_x = torch.unsqueeze(img_x, 0)

        out = model(img_x)
        _, _, high, width = out.shape
        out = out.detach().numpy()
        for h in range(0, high):
            for w in range(0, width):
                if out[0, 0, h, w] >= 0.5:
                    out[0, 0, h, w] = 1
                if out[0, 0, h, w] < 0.5:
                    out[0, 0, h, w] = 0
        trann = transforms.ToPILImage()
        out = torch.tensor(out)
        out = torch.squeeze(out)
        out = trann(out)
        '以原文件名保存到新的文件夹'
        out.save(test_config.out_path + pictureName)
        print("已输出保存: " + pictureName)
"""******************************************************************************************"""


if __name__ == "__main__":
    ' 函数入口：开始输出预测图'
    test_image()