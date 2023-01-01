
"""================================= 一、 引入相关库文件 ======================================"""
import os
' 多卡训练 '
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"    # 多卡训练
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"        # 单卡训练

from F_Others.indicators import *

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter  # 引入自带tensorboard库，用于记录损失函数和指标的变化
import torch.nn.functional as F
import os, argparse, math, sys
import logging
import torch.utils.data as data
from torch.nn import init
import cv2
# import visdom # 使用visdom记录指标和输出图像


' 引入自己的网络 '
from D_Model.Unet import Unet
"""******************************************************************************************"""

"""================================= 二、 初始化网络相关参数 =================================="""
parser = argparse.ArgumentParser()  # 建立解析对象

' 加载模型 '
parser.add_argument('--model', type=str, default=Unet(3, 1), help='所使用的网络模型:{Unet(3, 1); SwinUnet(img_size=224, num_classes=1)}')

' 数据集地址 '
parser.add_argument('--train_image', type=str,
                    default=r'./data/train/image', help='训练集原图路径')
parser.add_argument('--train_label', type=str,
                    default=r'./data/train/label', help='训练集标签路径')
parser.add_argument('--val_image', type=str,
                    default=r'./data/test/image', help='验证集原图路径')
parser.add_argument('--val_label', type=str,
                    default=r'./data/test/label', help='验证集标签路径')

' Tensorboard输出保存地址  一般取默认地址 '
parser.add_argument('--all_path', type=str, default='A_Indicators/Other/all', help='Tensorboard全部数据保存地址')
parser.add_argument('--train_path', type=str, default='A_Indicators/Other/train', help='Tensorboard训练集数据保存地址')
parser.add_argument('--val_path', type=str, default='A_Indicators/Other/val', help='Tensorboard验证集数据保存地址')
parser.add_argument('--weight', type=str, default='B_Weight/Other/', help='权重保存地址')
parser.add_argument('--logs', type=str, default='C_Logs/indicators_Unet.log', help='日志文件保存地址')
parser.add_argument('--logs_name', type=str, default='Net: Unet', help='日志文件名称')
parser.add_argument('--out_path', type=str, default=r"G_Out_hot_image/Other", help='预测热图输出路径')


' 设置其他参数 '
parser.add_argument('--device', type=str, default='cuda:1', help='使用的GPU编号: {cpu; cuda:0; cuda:1; "cuda" if torch.cuda.is_available() else "cpu"}')
parser.add_argument('--epochs', type=int, default=400, help='最大训练轮数')
parser.add_argument('--batchs', type=int, default=4, help='batch size')
parser.add_argument('--image_size', type=int, default=224, help='输入图片尺寸')
parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
parser.add_argument('--testcriterion', type=str, default=nn.BCELoss(), help='损失函数类型: {nn.BCELoss(); nn.BCEWithLogitsLoss(); nn.CrossEntropyLoss()}')
parser.add_argument('--init_means', type=str, default='kaiming', help='模型初始化方法: {normal; xavier; kaiming; orthogonal}')


parse_config = parser.parse_args()
# print(parse_config)   # 用于查看变量
"""******************************************************************************************"""

"""==================================== 三、 模型初始化方法 ======================================"""
def init_weights(net, init_type=parse_config.init_means, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
"""******************************************************************************************"""


"""==================================== 四、 读取数据集 ======================================"""
' 读取数据，将原图与标签进行对应组合 '
def photoname_dataset(root1, root2):   # 传入两个路径（原图 和 label）
    imgs = []
    photoname1 = os.listdir(root1)  # 获取该文件夹下图片总数
    photoname2 = os.listdir(root2)  # 获取该文件夹下图片总数
    n = len(os.listdir(root1))
    print("Picture number", n)
    img_list = []
    label_list = []
    for name1 in photoname1:  # 循环读取文件夹中的每个图片，建议对照着文件夹中的图片名称阅读此代码
        img = os.path.join(root1, name1)  # img是个变量，只能存储一个
        img_list.append(img)
    for name2 in photoname2:  # 循环读取文件夹中的每个图片，建议对照着文件夹中的图片名称阅读此代码
        label = os.path.join(root2, name2)
        label_list.append(label)
    n = len(os.listdir(root1))  # 获取该文件夹下图片总数
    for i in range(n):
        imgs.append((img_list[i], label_list[i]))  # 将读取到的原图+label一一对应，存储在imgs中
    return imgs

' 加载训练数据 '
class TrainDataset(data.Dataset):
    def __init__(self, root1, root2, transform=None, target_transform=None):
        imgs = photoname_dataset(root1, root2)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        img_x = img_x.resize((parse_config.image_size, parse_config.image_size))
        img_y = img_y.resize((parse_config.image_size, parse_config.image_size), Image.NEAREST)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

' 加载验证数据 '
class ValDataset(data.Dataset):
    def __init__(self, root1, root2, transform=None, target_transform=None):
        imgs = photoname_dataset(root1, root2)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        img_x = img_x.resize((parse_config.image_size, parse_config.image_size))
        img_y = img_y.resize((parse_config.image_size, parse_config.image_size), Image.NEAREST)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y
    def __len__(self):
        return len(self.imgs)

' 获取验证数据的路径名称 '
val_path = parse_config.val_image
Filelist = os.listdir(val_path)
"""******************************************************************************************"""

"""===================================== 五、 前期准备工作 ==================================="""
' 将训练数据记录到Tensorboard中 '
wirter_all = SummaryWriter(parse_config.all_path)      # 全部数据
writer_train = SummaryWriter(parse_config.train_path)  # 训练数据
writer_val = SummaryWriter(parse_config.val_path)      # 验证数据

' 所使用的设备 '
USE_CUDA = torch.cuda.is_available()
device = torch.device(parse_config.device if USE_CUDA else "cpu")
torch.backends.cudnn.benchmark = True

' 对（inputs）输入图片的数据进行操作 '
x_transforms = transforms.Compose([
    transforms.ToTensor(),      # 将数据转为tensor类型，方便pytorch进行自动求导，优化之类的操作
    # transforms.Lambda(lambda x: x.repeat(3,1,1)),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 数据归一化，两个参数，一个为均值，一个为方差，均设置为0.5，每个参数里三个0.5表示有三个通道
    transforms.Normalize([0.5], [0.5])  # 单通道
])

' 对（label图）标签图只需要转换为tensor '
y_transforms = transforms.ToTensor()    # 将数据转为tensor类型，方便pytorch进行自动求导，优化之类的操作

' 创建日志 '
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

' 生成热图时对网络最后一层输出以及label的处理 '
def Normalize(data):
    mx = np.max(data)
    mn = np.min(data)
    delta = 1e-10
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = ((data[i, j]-mn) + delta) / ((mx-mn) + delta)
    return data

def show(data):
    data = Normalize(data)*255
    jet = np.uint8(data)
    jet = cv2.applyColorMap(jet, cv2.COLORMAP_JET)
    return jet

def fuse(out, label):
    out = torch.squeeze(out).cpu().detach().numpy()

    label = torch.squeeze(label).cpu().detach().numpy()*255
    # label = cv2.applyColorMap(label, cv2.COLORMAP_JET)
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)

    out = show(out)
    label = show(label)

    H = parse_config.image_size
    W = parse_config.image_size

    result = np.zeros((H, 2*W, 3), np.float32)
    result[0:H, 0:W, :] = out
    result[0:H, W:2*W, :] = label

    return result
"""******************************************************************************************"""

"""===================================== 六、 定义训练模型 ==================================="""
def train_model(model, criterion, dataload, num_epochs=parse_config.epochs):
    #           模型   损失函数类型  优化器      数据     训练轮数（可更改）
    epoch_num = 0   # 记录epoch数

    ' 创建日志 '
    best_Precision = 0
    best_Sensitivity = 0
    best_Dice = 0
    best_Accuracy = 0
    best_IoU = 0
    best_Specificity = 0
    best_FPR = 100
    best_epoch = 0
    logger = get_logger(parse_config.logs)
    logger.info(parse_config.logs_name)

    ' Visdom 相关设置'
    # vis = visdom.Visdom(env='index')   # vis = visdom.Visdom(env='develop')   其中develop为创建的Enviroment名称
    # vis_img = visdom.Visdom(env='hot_image')   # vis = visdom.Visdom(env='develop')   其中develop为创建的Enviroment名称

    """ =====================设置动态学习率========================= """
    ' 选择参数调整方式 '
    lambda1 = lambda epoch: epoch // 5  # 第一组参数的调整方法
    lambda2 = lambda epoch: 0.98 ** epoch  # 第二组参数的调整方法

    ' 优化器选择 '
    optimizer = optim.Adam(model.parameters(), lr=parse_config.lr)  # 选择Adam优化器
    # optimizer = optim.AdamW(model.parameters(), lr=parse_config.lr)  # 选择AdamW优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=parse_config.lr, momentum=0.9)  # 选择SGD优化器

    ' 学习率的调度器设置 '
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2, last_epoch=-1)  # 根据函数lr_lambda改变学习率
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)    # 在每step_size个epoch时，学习率乘以gamma值
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1, last_epoch=-1)  # 在第10、20、30epoch时调整
    """ *********************************************************** """

    for epoch in range(num_epochs):
           #训练开始
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        dt_size = len(dataload.dataset)
        epoch_num += 1
        epoch_loss = 0
        step = 0

        PrecisionT = 0.0
        SensitivityT = 0.0
        DiceT = 0.0
        AccuracyT = 0.0
        IoUT = 0.0
        SpecificityT = 0.0
        FPRT = 0.0

        CM_total = np.zeros((2, 2))

        for x, y in dataload:
            step += 1

            inputs = x.to(device)   # 原图
            labels = y.to(device)   # 标签图

            optimizer.zero_grad()  # zero the parameter gradients 参数梯度归零
            outputs = model(inputs)   # 前向传播  经过网络输出预测图

            loss = criterion(outputs, labels)  # 计算loss
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度下降

            """ ======================= 指标计算 ==========================="""
            Dic = Dice(outputs.to("cpu"), labels.to("cpu"))
            Sen = SE(outputs.to("cpu"), labels.to("cpu"))
            Iou = IOU(outputs.to("cpu"), labels.to("cpu"))
            Pre = Precision(outputs.to("cpu"), labels.to("cpu"))
            Acc = ACC(outputs.to("cpu"), labels.to("cpu"))
            Spe = SP(outputs.to("cpu"), labels.to("cpu"))
            Fpr = FPR(outputs.to("cpu"), labels.to("cpu"))

            PrecisionT += Pre
            SensitivityT += Sen
            DiceT += Dic
            AccuracyT += Acc
            IoUT += Iou
            SpecificityT += Spe
            FPRT += Fpr

            epoch_loss += loss.item()    # 累计Loss值
            """ ==========================================================="""

            print("Training... epoch:%d, %d/%d, Loss:%0.3f, Precision:%0.3f, Sensitivity:%0.3f, Dice:%0.3f, Accuracy:%0.3f, IoU:%0.3f, Specificity:%0.3f, FPR:%0.3f"
                  % (epoch+1, step, (dt_size-1) // dataload.batch_size+1, loss.item(), Pre, Sen, Dic, Acc, Iou, Spe, Fpr))

        scheduler.step()  # 需要在优化器参数更新之后再动态调整学习率，每次更新是根据epoch数量在两个函数上计算得到的乘数因子进行更新

        ' 计算 epoch lmodel.train()oss 值 '
        train_Loss = (epoch_loss / ((dt_size - 1) / dataload.batch_size + 1))

        epoch_Precision = (PrecisionT / ((dt_size - 1) / dataload.batch_size + 1))
        epoch_Sensitivity = (SensitivityT / ((dt_size - 1) / dataload.batch_size + 1))
        epoch_Dice = (DiceT / ((dt_size - 1) / dataload.batch_size + 1))
        epoch_Accuracy = (AccuracyT / ((dt_size - 1) / dataload.batch_size + 1))
        epoch_IoU = (IoUT / ((dt_size - 1) / dataload.batch_size + 1))
        epoch_Specificity = (SpecificityT / ((dt_size - 1) / dataload.batch_size + 1))
        epoch_FPR = (FPRT / ((dt_size - 1) / dataload.batch_size + 1))

        ' 将 当前训练 的Loss、IoU、Dice、Acc等数据写入Tensorboard '
        writer_train.add_scalar("train_Loss", train_Loss, epoch_num)
        writer_train.add_scalar("train_Precision", epoch_Precision, epoch_num)
        writer_train.add_scalar("train_Sensitivity", epoch_Sensitivity, epoch_num)
        writer_train.add_scalar("train_Dice", epoch_Dice, epoch_num)
        writer_train.add_scalar("train_Accuracy", epoch_Accuracy, epoch_num)
        writer_train.add_scalar("train_IoU", epoch_IoU, epoch_num)
        writer_train.add_scalar("train_Specificity", epoch_Specificity, epoch_num)
        writer_train.add_scalar("train_FPR", epoch_FPR, epoch_num)

        ' 输出提示信息 Precision, Sensitivity, Dice, Accuracy, IoU, Specificity, FPR '
        print('******** Epoch: %d 结果 ********' % (epoch+1))
        print(" Loss: %0.3f" % train_Loss)
        print(" Precision: %0.3f" % epoch_Precision)
        print(" Sensitivity: %0.3f" % epoch_Sensitivity)
        print(" Dice: %0.3f" % epoch_Dice)
        print(" Accuracy: %0.3f" % epoch_Accuracy)
        print(" IoU: %0.3f" % epoch_IoU)
        print(" Specificity: %0.3f" % epoch_Specificity)
        print(" FPR: %0.3f" % epoch_FPR)
        print('*' * 30)

        """====== 测试 ======"""
        model.eval()
        val_batch = 1

        ' 损失函数 '
        Testcriterion = parse_config.testcriterion

        Test_dataset = ValDataset(parse_config.val_image, parse_config.val_label, transform=x_transforms, target_transform=y_transforms)
        Testdataloaders = DataLoader(Test_dataset, batch_size=val_batch, shuffle=True, num_workers=7)
        with torch.no_grad():
            val_Loss, val_Precision, val_Sensitivity, val_Dice, val_Accuracy, val_IoU, val_Specificity, val_FPR, outimage, outlabel = test_model(model, Testcriterion, Testdataloaders, epoch_num=epoch_num)

        ' 将 训练和测试 的Loss、IoU、Dice、Acc等数据写入 Tensorboard '
        wirter_all.add_scalars("Loss", {'train_loss': train_Loss, 'val_loss': val_Loss}, epoch_num)
        wirter_all.add_scalars("Precision", {'train_Precision': epoch_Precision, 'val_Precision': val_Precision}, epoch_num)
        wirter_all.add_scalars("Sensitivity", {'train_Sensitivity': epoch_Sensitivity, 'val_Sensitivity': val_Sensitivity}, epoch_num)
        wirter_all.add_scalars("Dice", {'train_Dice': epoch_Dice, 'val_Dice': val_Dice}, epoch_num)
        wirter_all.add_scalars("Accuracy", {'train_Accuracy': epoch_Accuracy, 'val_Accuracy': val_Accuracy}, epoch_num)
        wirter_all.add_scalars("IoU", {'train_IoU': epoch_IoU, 'val_IoU': val_IoU}, epoch_num)
        wirter_all.add_scalars("Specificity", {'train_Specificity': epoch_Specificity, 'val_Specificity': val_Specificity}, epoch_num)
        wirter_all.add_scalars("FPR", {'train_FPR': epoch_FPR, 'val_FPR': val_FPR}, epoch_num)

        ' 存储模型参数，后面引号里面的是存储位置 '
        torch.save(model.state_dict(), parse_config.weight+'weights_%d_%0.3f.pth' % (epoch+1, val_Dice))

        if best_Dice < val_Dice:
            best_Dice = val_Dice
            best_epoch = epoch_num
        if best_Precision < val_Precision:
            best_Precision = val_Precision
        if best_Sensitivity < val_Sensitivity:
            best_Sensitivity = val_Sensitivity
        if best_Accuracy < val_Accuracy:
            best_Accuracy = val_Accuracy
        if best_IoU < val_IoU:
            best_IoU = val_IoU
        if best_Specificity < val_Specificity:
            best_Specificity = val_Specificity
        if best_FPR > val_FPR:
            best_FPR = val_FPR

        ' 将 训练和测试 的Loss、IoU、Dice、Acc等数据写入 日志文件 '
        logger.info('Epoch:[{}/{}]\t val_Loss={:.5f}\t val_Precision={:.3f}\t val_Sensitivity={:.3f}\t val_Dice={:.3f}\t val_Accuracy={:.3f}\t val_IoU={:.3f}\t val_Specificity={:.3f}\t val_FPR={:.3f}\t'
                    .format(epoch+1, num_epochs, val_Loss, val_Precision, val_Sensitivity, val_Dice, val_Accuracy, val_IoU, val_Specificity, val_FPR))
        logger.info('best_Epoch={}\t best_Precision={:.3f}\t best_Sensitivity={:.3f}\t best_Dice={:.3f}\t best_Accuracy={:.3f}\t best_IoU={:.3f}\t best_Specificity={:.3f}\t  best_FPR={:.3f}\t'
                    .format(best_epoch, best_Precision, best_Sensitivity, best_Dice, best_Accuracy, best_IoU, best_Specificity, best_FPR))
"""******************************************************************************************"""

"""===================================== 七、 定义验证模型 ==================================="""
def test_model(model, criterion, dataload, epoch_num, num_epochs=1):
    with torch.no_grad():
        print(' Testing... ')
        for epoch in range(num_epochs):
            dt_size = len(dataload.dataset)
            epoch_loss = 0
            step = 0

            PrecisionT = 0.0
            SensitivityT = 0.0
            DiceT = 0.0
            AccuracyT = 0.0
            IoUT = 0.0
            SpecificityT = 0.0
            FPRT = 0.0

            CM_total = np.zeros((2, 2))

            for x, y in dataload:
                step += 1
                inputs = x.to(device)
                labels = y.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)


                """ ======================= 指标计算 ==========================="""
                Dic = Dice(outputs.to("cpu"), labels.to("cpu"))
                Sen = SE(outputs.to("cpu"), labels.to("cpu"))
                Iou = IOU(outputs.to("cpu"), labels.to("cpu"))
                Pre = Precision(outputs.to("cpu"), labels.to("cpu"))

                Acc = ACC(outputs.to("cpu"), labels.to("cpu"))
                Spe = SP(outputs.to("cpu"), labels.to("cpu"))
                Fpr = FPR(outputs.to("cpu"), labels.to("cpu"))

                PrecisionT += Pre
                SensitivityT += Sen
                DiceT += Dic
                AccuracyT += Acc
                IoUT += Iou
                SpecificityT += Spe
                FPRT += Fpr

                epoch_loss += loss.item()  # 累计Loss值
                """ ==========================================================="""

                """ ======================= 生成热力图 ======================== """
                if epoch_num > 5:
                    if step % 10 == 0:
                        CAM = fuse(outputs, labels)
                        cv2.imwrite(parse_config.out_path + '/{}_'.format(epoch_num) + Filelist[step-1], CAM)
                """ ********************************************************** """

            test_Loss = (epoch_loss / ((dt_size - 1) // dataload.batch_size + 1))


            epoch_Precision = (PrecisionT / ((dt_size - 1) / dataload.batch_size + 1))
            epoch_Sensitivity = (SensitivityT / ((dt_size - 1) / dataload.batch_size + 1))
            epoch_Dice = (DiceT / ((dt_size - 1) / dataload.batch_size + 1))
            epoch_Accuracy = (AccuracyT / ((dt_size - 1) / dataload.batch_size + 1))
            epoch_IoU = (IoUT / ((dt_size - 1) / dataload.batch_size + 1))
            epoch_Specificity = (SpecificityT / ((dt_size - 1) / dataload.batch_size + 1))
            epoch_FPR = (FPRT / ((dt_size - 1) / dataload.batch_size + 1))

            ' 将 当前测试 的Loss、IoU、Dice、Acc等数据写入tensorboard '
            writer_val.add_scalar("test_Loss", test_Loss, epoch_num)
            writer_val.add_scalar("test_Precision", epoch_Precision, epoch_num)
            writer_val.add_scalar("test_Sensitivity", epoch_Sensitivity, epoch_num)
            writer_val.add_scalar("test_Dice", epoch_Dice, epoch_num)
            writer_val.add_scalar("test_Accuracy", epoch_Accuracy, epoch_num)
            writer_val.add_scalar("test_IoU", epoch_IoU, epoch_num)
            writer_val.add_scalar("test_Specificity", epoch_Specificity, epoch_num)
            writer_val.add_scalar("test_FPR", epoch_FPR, epoch_num)

            ' 输出提示信息 Precision, Sensitivity, Dice, Accuracy, IoU, Specificity, FPR '
            print('******** Test 结果 ********')
            print(" Loss: %0.3f" % test_Loss)
            print(" Precision: %0.3f" % epoch_Precision)
            print(" Sensitivity: %0.3f" % epoch_Sensitivity)
            print(" Dice: %0.3f" % epoch_Dice)
            print(" Accuracy: %0.3f" % epoch_Accuracy)
            print(" IoU: %0.3f" % epoch_IoU)
            print(" Specificity: %0.3f" % epoch_Specificity)
            print(" FPR: %0.3f" % epoch_FPR)
            print('*' * 30)
        return test_Loss, epoch_Precision, epoch_Sensitivity, epoch_Dice, epoch_Accuracy, epoch_IoU, epoch_Specificity, epoch_FPR, outputs, labels
"""******************************************************************************************"""

"""===================================== 八、 完善训练模型==================================="""
def train(batch_size):
    ' 单卡训练 '
    model = parse_config.model.to(device)
    ' 多卡训练 '
    # model = torch.nn.DataParallel(parse_config.model, device_ids=[1, 0])
    # model.to(device)

    ' 模型初始化 '
    model.apply(init_weights)

    '''损失函数的设置'''
    criterion = parse_config.testcriterion

    '''训练数据集加载与处理'''
    Train_dataset = TrainDataset(parse_config.train_image, parse_config.train_label, transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(Train_dataset, batch_size=batch_size, shuffle=True, num_workers=7)
    train_model(model, criterion, dataloaders)
"""******************************************************************************************"""


if __name__ == "__main__":
    ' 函数入口：开始训练网络'
    train(parse_config.batchs)