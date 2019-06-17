# 用MTCNN进行人脸追踪

第一步：使用CelebA或者其他数据集制作训练数据集：data_sample.py
数据集分为三个部分，分别是positive：正样本、negative：负样本、part：部分样本。
三个网络都要有这三个训练样本，训练样本起着至关重要的作用，直接决定网络能否训练好，尤其是负样本。

第二步：对训练数据进行采样：DataSet.py
第三步：构建网络：Net.py


![images](https://github.com/lihuaqiang0101/MTCNN-Face-Detection-System/blob/master/images/net.png)


网络分为三个，P网络主要是生成可能的建议框，R网络负责筛选P网络产生的建议框，O网络再对R网产生的建议框做进一步的筛选。
第四步：对网络进行训练：train.py
第五步：对训练好的网络进行测试：mtcnntest.py、mtcnnvediotest.py
在测试过程中的流程如下：
P网络：
对测试图片做图像金字塔：

![images](https://github.com/lihuaqiang0101/MTCNN-Face-Detection-System/blob/master/images/%E5%9B%BE%E5%83%8F%E9%87%91%E5%AD%97%E5%A1%94.png)


scale = 1
while min_len > 12:
  scale *= 0.7
   _w = w * scale
   _h = h * scale
  img = image.resize((_w,_h))
  min_len = min(_w,_h)
对图像金字塔中的每一层特征层传入P网络获得每个像素点的置信度和偏移：
conf,off  = self.pnet(imgdata)
筛选出置信度较大的点作为建议框的起始点：
indexs = torch.nonzero(torch.gt(confidence,0.6))
将这些点映射回原图得到建议框：
    def get_boxs(self,star_index,confidence,offset,sclae,stride=2,side_len=2):
        _x1 = star_index[:1] * stride / sclae
        _y1 = star_index[:0] * stride / sclae
        _x2 = (star_index[:,1] * stride + side_len) / sclae
        _y2 = (star_index[:,0] * stride + side_len) / sclae
        w = _x2 - _x1
        h = _y2 - _y1
        x1 = _x1 + offset[:,star_index[:,1],0] * w
        y1 = _y1 + offset[:,star_index[:,0],1] * h
        x2 = _x2 + offset[:,star_index[:,1],2] * w
        y2 = _y2 + offset[:star_index[:,0],3] * h
        return [x1,y1,x2,y2,confidence[star_index]]
 对得出的框采用非极大抑制过滤掉每个对象置信度相对较高的框：
     def nms(self,boxs,theta,isMin):
        if boxs.shape[0] == 0:
            return np.array([])
        sort_boxs = boxs[(-boxs[:,4]).argsort()]
        bboxes = []
        while sort_boxs.shape[0] > 1:
            max_box = sort_boxs[0]
            bboxs = sort_boxs[1:]
            bboxes.append(max_box)
            indexs = np.where(self.iou(max_box,bboxs,isMin) < theta)
            sort_boxs = bboxs[indexs]
        if sort_boxs.shape[0] > 1:
            bboxes.append(sort_boxs)
        return torch.stack(bboxes)
  在非极大抑制中将置信度最大框与剩下的框做IOU：
  def iou(self,box,boxs,isMin):
        area = (box[2] - box[0]) * (box[3] - box[1])
        areas = (boxs[:,2] - boxs[:,0]) - (boxs[:,3] - boxs[:,1])
        x1 = np.minimum(box[0],boxs[:,0])
        y1 = np.minimum(box[1],boxs[:,1])
        x2 = np.maximum(box[2],boxs[:,2])
        y2 = np.maximum(box[3],boxs[:,3])
        w = x2 - x1
        h = y2 - y1
        integer = w * h
        if isMin:
            var = np.true_divide(integer,np.minimum(area,areas))
        else:
            var = np.true_divide(integer,areas + area - integer)
        return integer
   在R网中首先对p网络的框转换成正方形的：
       def convert_to_square(self,boxs):
        squareboxs = boxs.copy()
        w = boxs[:,2] - boxs[:,0]
        h = boxs[:,3] - boxs[:,1]
        max_side = max(w,h)
        squareboxs[:,0] = boxs[:,0] + w * 0.5 - max_side * 0.5
        squareboxs[:,1] = boxs[:,1] + h * 0.5 - max_side * 0.5
        squareboxs[:,2] = squareboxs[:,0] + max_side
        squareboxs[:,3] = squareboxs[:,3] + max_side
        return squareboxs
  然后根据这些框在原始图片上进行裁剪和缩放：
          for box in pnet_boxs:
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            img = image.crop((_x1,_y1,_x2,_y2))
            img = img.resize((24,24))
            imgdata = self.ToTensor(img)
            _imgdatasets.append(imgdata)
   将缩放后的图片传入R网得出每个框的置信度和偏移
   根据置信度对这些框进行筛选
   对选出来的框进行微调：
           for index in indexs:
            _x1 = int(pnet_boxs[index][0])
            _y1 = int(pnet_boxs[index][1])
            _x2 = int(pnet_boxs[index][2])
            _y2 = int(pnet_boxs[index][3])
            w = _x2 - _x1
            h = _y2 - _y1
            x1 = _x1 + offset[index][0] * w
            y1 = _y1 + offset[index][1] * h
            x2 = _x2 + offset[index][2] * w
            y2 = _y2 + offset[index][3] * h
            rnetboxs.append([x1,y1,x2,y2,confidence[index][0]])
  做非极大抑制得出最终的框
  O网络与R网络的流程是相同的，只不过在缩放图片的时候是将24x24的改为了48x48
