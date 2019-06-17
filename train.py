import torch
from torch import nn
from torch import optim
from DataSet import MyDataSet
from Net import Pnet,Rnet,Onet
import DataSet
from torch.utils import data

class Train:
    def __init__(self,p_textpath,n_textpath,t_textpath,p_imgpath,n_imgpath,t_imgpath,net):
        self.p_textpath = p_textpath
        self.p_imgpath = p_imgpath
        self.n_textpath = n_textpath
        self.n_imgpath = n_imgpath
        self.t_textpath = t_textpath
        self.t_imgpath = t_imgpath
        self.net = net
        #创建训练数据集
        dataset = MyDataSet(p_textpath,n_textpath,t_textpath,p_imgpath,n_imgpath,t_imgpath)
        self.dataloader = data.DataLoader(dataset,batch_size=100,shuffle=True)

    def train(self,save_path):
        if self.net == 'pnet':
            net = Pnet()
        elif self.net == 'rnet':
            net = Rnet()
        elif self.net == 'onet':
            net = Onet()
        net = net.cuda()
        # net.load_state_dict(torch.load(save_path))
        optimizer = optim.Adam(net.parameters())
        conf_loss_fun = nn.BCELoss()
        off_loos_fun = nn.MSELoss()
        for epoch in range(20000):
            imgdata, conf, offset = DataSet.GetIter(self.dataloader)
            imgdata = imgdata.cuda()
            conf = conf.cuda()
            offset = offset.cuda()
            confidence,offset_out = net(imgdata)
            #置信度的损失需要正负样本
            #获得置信度小于2的掩码
            conn_mask = torch.lt(conf,2)
            #得到符合条件的置信度
            conf_ = conf[conn_mask]
            confidence_ = confidence[conn_mask]
            #偏移的损失需要正样本和部分样本
            #得到置信度大于0的掩码
            off_mask = torch.gt(conf,0)
            #得到符合条件的偏移
            offset = offset[off_mask[:,0]]
            offset_out = offset_out[off_mask[:,0]]
            conf_loss = conf_loss_fun(confidence_,conf_)
            off_loss = off_loos_fun(offset_out,offset)
            loss = conf_loss + off_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
        torch.save(net.state_dict(),save_path)

pnet = Train(DataSet.p_12txtpath,DataSet.n_12txtpath,DataSet.t_12txtpath,DataSet.p_12imgpath,DataSet.n_12imgpath,DataSet.t_12imgpath,'pnet')
pnet.train(r'ppra\pnet_param.pkl')
rnet = Train(DataSet.p_24txtpath,DataSet.n_24txtpath,DataSet.t_24txtpath,DataSet.p_24imgpath,DataSet.n_24imgpath,DataSet.t_24imgpath,'rnet')
rnet.train(r'ppra\rnet_param.pkl')
onet = Train(DataSet.p_48txtpath,DataSet.n_48txtpath,DataSet.t_48txtpath,DataSet.p_48imgpath,DataSet.n_48imgpath,DataSet.t_48imgpath,'onet')
onet.train(r'ppra\onet_param.pkl')