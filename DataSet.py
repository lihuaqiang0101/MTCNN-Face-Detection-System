p_12txtpath = r"D:\CelebA\data\12\positive.txt"
n_12txtpath = r"D:\CelebA\data\12\negative.txt"
t_12txtpath = r"D:\CelebA\data\12\part.txt"
p_24txtpath = r"D:\CelebA\data\24\positive.txt"
n_24txtpath = r"D:\CelebA\data\24\negative.txt"
t_24txtpath = r"D:\CelebA\data\24\part.txt"
p_48txtpath = r"D:\CelebA\data\48\positive.txt"
n_48txtpath = r"D:\CelebA\data\48\negative.txt"
t_48txtpath = r"D:\CelebA\data\48\part.txt"
p_12imgpath = r"D:\CelebA\data\12"
n_12imgpath = r"D:\CelebA\data\12\negative"
t_12imgpath = r"D:\CelebA\data\12"
p_24imgpath = r"D:\CelebA\data\24"
n_24imgpath = r"D:\CelebA\data\24\negative"
t_24imgpath = r"D:\CelebA\data\24"
p_48imgpath = r"D:\CelebA\data\48"
n_48imgpath = r"D:\CelebA\data\48\negative"
t_48imgpath = r"D:\CelebA\data\48"

from torch.utils import data
import numpy as np
from PIL import Image
import os
import torch

class MyDataSet(data.Dataset):
    def __init__(self,p_path, n_path, t_path, p_imgpath, n_imgpath, t_imgpath):
        super(MyDataSet, self).__init__()
        #训练数据集的路径
        #标签的路径
        self.p_path = p_path
        self.n_path = n_path
        self.t_path = t_path
        #图片的路径
        self.p_imgpath = p_imgpath
        self.n_imgpath = n_imgpath
        self.t_imgpath = t_imgpath
        #读取相应的标签文件
        p_file = open(p_path,'r')
        n_file = open(n_path,'r')
        t_file = open(t_path,'r')
        pdata = p_file.readlines()
        ndata = n_file.readlines()
        tdata = t_file.readlines()
        #将正样本、负样本、部分样本按照3：9：3的比例进行采样作为训练数据集
        self.dataset = []
        self.dataset.extend(np.random.choice(pdata,size=3000))#np.random.choice随机选取一个列表中的size个元素组成一个新的列表
        self.dataset.extend(np.random.choice(ndata,size=9000))
        self.dataset.extend(np.random.choice(tdata,size=3000))

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        #将原始的文本变为数据的列表
        strs = self.dataset[index].strip().split(' ')
        #判断读取到的是哪一类样本并进行相应的读取
        if strs[1] == '0':#负样本
            #读取图片并做归一化
            imagedata = np.array(Image.open(os.path.join(self.n_imgpath,strs[0])),dtype=np.float32)/255
        elif strs[1] == '1':#正样本
            imagedata = np.array(Image.open(os.path.join(self.p_imgpath,strs[0])),dtype=np.float32)/255
        elif strs[1] == '2':
            imagedata = np.array(Image.open(os.path.join(self.t_imgpath,strs[0])),dtype=np.float32)/255
        #将图片转换为torch的CHW的形式
        imagedata = np.transpose(imagedata,[2,0,1])
        #将要训练的数据转换为torch的tensor类型
        imagedata = torch.FloatTensor(imagedata)#图片
        confidence = torch.FloatTensor(np.array([float(strs[1])]))#置信度
        offest = torch.FloatTensor(np.array([float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])]))#偏移
        return imagedata,confidence,offest

# Data = MyDataSet(p_12txtpath,n_12txtpath,t_12txtpath,p_12imgpath,n_12imgpath,t_12imgpath)
def GetIter(dataloader):
    #将数据集加载过来,batch_size表示只取数据集中batch_size个数据，使用batch_size不能超过数据集的个数
    iters = iter(dataloader)#将加载过来的数据构造成一个迭代器
    #对数据进行迭代相当于执行了getitem
    imgdata,conf,offset = iters.next()
    return imgdata,conf,offset