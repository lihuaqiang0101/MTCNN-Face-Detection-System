# 用MTCNN进行人脸追踪

第一步：使用CelebA或者其他数据集制作训练数据集：data_sample.py
数据集分为三个部分，分别是positive：正样本、negative：负样本、part：部分样本。
三个网络都要有这三个训练样本。

第二步：对训练数据进行采样：DataSet.py
第三步：构建网络：Net.py


![images](https://github.com/lihuaqiang0101/MTCNN-Face-Detection-System/blob/master/images/net.png)


