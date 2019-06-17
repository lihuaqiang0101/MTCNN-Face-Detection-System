import torch
from Net import Pnet,Rnet,Onet
from utils import PnetDetect,RnetDetect,OnetDetect
from PIL import Image
import cv2

if __name__ == '__main__':
    pnet = Pnet()
    rnet = Rnet()
    onet = Onet()
    if torch.cuda.is_available():
        pnet = pnet.cuda()
        rnet = rnet.cuda()
        onet = onet.cuda()
    pnet.load_state_dict(torch.load(r'C:\Users\34801\Desktop\MTCNN人脸检测\para\pnet_param.pkl'))
    rnet.load_state_dict(torch.load(r'C:\Users\34801\Desktop\MTCNN人脸检测\para\rnet_param.pkl'))
    onet.load_state_dict(torch.load(r'C:\Users\34801\Desktop\MTCNN人脸检测\para\onet_param.pkl'))
    img = Image.open(r'a.jpg')
    pbox = PnetDetect(pnet, img, imgshow=False)
    rbox = RnetDetect(rnet, img=img, boxs=pbox, imgshow=False)
    obox =OnetDetect(onet,img=img,boxs=rbox,imgshow=True)
    cv2.waitKey(0)