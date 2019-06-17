import os
import numpy as np
from utils import IouDo
from PIL import Image
from PIL import ImageFilter

path = r'D:\CelebA'
img_path = r"D:\CelebA\img_celeba"
label_path = r"D:\CelebA\list_bbox_celeba.txt"

#创建用于存放相应尺寸训练样本的文件夹，里面分别有对应的正样本、负样本和部分样本
def makedir(size):
    root_path = os.path.join(path,str(size))
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    p_path = os.path.join(root_path,'positive')
    if not os.path.exists(p_path):
        os.mkdir(p_path)
    n_path = os.path.join(root_path,'negative')
    if not os.path.exists(n_path):
        os.mkdir(n_path)
    t_path = os.path.join(root_path,'part')
    if not os.path.exists(t_path):
        os.mkdir(t_path)
    return root_path,p_path,n_path,t_path

def create_traindata(size):
    imgcount = 0  # 用于计数，记录生成了多少张图片
    r_path, p_path, n_path, t_path = makedir(size)  # 生成目录
    # 生成保存标签的文件
    p_file = open(r_path + '/positive.txt', 'w')
    n_file = open(r_path + '/negative.txt', 'w')
    t_file = open(r_path + '/part.txt', 'w')
    # 读取标签中的每一行
    for index, lines in enumerate(open(label_path).readlines()):
        # 跳过前两行
        if index < 2:
            continue
        # 去除空格并进行切分以获取位置信息
        strs = lines.strip().split(' ')
        # 对不需要的信息进行过滤
        strs = list(filter(bool, strs))
        filename = strs[0]  # 获得图片名
        # 获取坐标信息
        x1 = int(strs[1])
        y1 = int(strs[2])
        w = int(strs[3])
        h = int(strs[4])
        x2 = x1 + w
        y2 = y1 + h
        # 计算中心点坐标
        cx = x1 + w // 2
        cy = y1 + h // 2
        # 获取w,h的最大值作为边界
        side = np.maximum(w, h)
        # 读取对应的图片
        img = Image.open(os.path.join(img_path, filename))
        # 获取原始图片的宽和高方便以后生成负样本
        width, high = img.size
        # 循环生成训练样本
        for count in range(5):
            # 生成随机的偏移
            offset_side = np.random.uniform(-0.2, 0.2) * side  # 框大小的偏移
            # 位置的偏移
            offset_x = np.random.uniform(-0.2, 0.2) * w / 2
            offset_y = np.random.uniform(-.02, 0.2) * h / 2
            # 偏移后的中心坐标
            _cx = cx + offset_x
            _cy = cy + offset_y
            # 偏移后的边界框宽度
            _side = side + offset_side
            # 偏移后的起始坐标(要防止越界)
            _x1 = np.maximum(_cx - _side * 0.5, 0)
            _y1 = np.maximum(_cy - _side * 0.5, 0)
            _x2 = _x1 + _side
            _y2 = _y1 + _side
            # 计算偏移
            offset_x1 = (x1 - _x1) / _side
            offset_y1 = (y1 - _y1) / _side
            offset_x2 = (x2 - _x2) / _side
            offset_y2 = (y2 - _y2) / _side
            # 计算IOU
            # 实际框
            box = np.array([x1, y1, x2, y2, 0])
            boxs = np.array([[_x1, _y1, _x2, _y2, 0]])
            # 置信度
            per = IouDo(box, boxs)
            # 获取IOU的数值
            per = per[0]
            # 裁剪出框所对应的图片
            tempimg = img.crop((_x1, _y1, _x2, _y2))
            # 将裁剪出的图片缩放成对应的大小并设置为抗锯齿防止因缩放造成过多的信息丢失
            tempimg = tempimg.resize((size, size), Image.ANTIALIAS)
            # 创建一个列表用于保存原始图片和经过模糊处理后的图片
            imglist = []
            imglist.append(tempimg)
            # 对图片进行模糊处理
            filterimg = tempimg.filter(ImageFilter.BLUR)  # 模糊滤波
            imglist.append(filterimg)
            for _tempimg in imglist:
                if per >= 0.65:
                    _tempimg.save('{0}/{1}.jpg'.format(p_path, imgcount))
                    p_file.write(
                        '{0}.jpg 1 {1} {2} {3} {4} 1\n'.format(imgcount, offset_x1, offset_y1, offset_x2, offset_y2))
                    imgcount += 1
                elif (per > 0.4) and (per < 0.65):
                    _tempimg.save('{0}/{1}.jpg'.format(t_path, imgcount))
                    t_file.write(
                        '{0}.jpg 2 {1} {2} {3} {4} 2\n'.format(imgcount, offset_x1, offset_y1, offset_x2, offset_y2))
                    imgcount += 1
        for count in range(10):
            offset_side = np.random.uniform(-0.2, 0.2) * side
            _side = side + offset_side
            # 裁剪负样本的起始坐标
            _x1 = np.random.uniform(0, width - _side)
            _y1 = np.random.uniform(0, high - _side)
            _x2 = _x1 + _side
            _y2 = _y1 + _side
            box = np.array([x1, y1, x2, y2, 0])
            boxs = np.array([[_x1, _y1, _x2, _y2, 0]])
            per = IouDo(box, boxs)
            per = per[0]
            offset_x1 = (x1 - _x1) / _side
            offset_y1 = (y1 - _y1) / _side
            offset_x2 = (x2 - _x2) / _side
            offset_y2 = (y2 - _y2) / _side
            if per < 0.3:
                tempimg = img.crop((_x1, _y1, _x2, _y2))
                tempimg = tempimg.resize((size, size), Image.ANTIALIAS)
                imglist = []
                imglist.append(tempimg)
                filterimg = tempimg.filter(ImageFilter.BLUR)
                imglist.append(filterimg)
                for _img in imglist:
                    _img.save('{0}/{1}.jpg'.format(n_path, imgcount))
                    n_file.write(
                        '{0}.jpg 0 {1} {2} {3} {4} 0\n'.format(imgcount, offset_x1, offset_y1, offset_x2, offset_y2))
                    imgcount += 1