import cv2
from Net import Pnet,Rnet,Onet
from utils import PnetDetect,RnetDetect,OnetDetect
import torch
from PIL import Image
import numpy as np

cap = cv2.VideoCapture(r'1.mp4')
pnet = Pnet()
rnet = Rnet()
onet = Onet()
if torch.cuda.is_available():
    pnet = pnet.cuda()
    rnet = rnet.cuda()
    onet = onet.cuda()
pnet.load_state_dict(torch.load(r'C:\Users\34801\Desktop\MTCNN人脸检测\params\pnet_param.pkl'))
rnet.load_state_dict(torch.load(r'C:\Users\34801\Desktop\MTCNN人脸检测\params\rnet_param.pkl'))
onet.load_state_dict(torch.load(r'C:\Users\34801\Desktop\MTCNN人脸检测\params\onet_param.pkl'))
while cap.isOpened():
    _,fram = cap.read()
    b,g,r = cv2.split(fram)
    img = cv2.merge([r,g,b])
    img = Image.fromarray(img.astype(np.uint8))
    pnetboxs = PnetDetect(img=img,imgshow=False,net=pnet)
    rnetboxs = RnetDetect(rnet,img=img,boxs=pnetboxs)
    onetboxs = OnetDetect(onet,img,rnetboxs,imgshow=True)