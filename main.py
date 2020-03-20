import torch as t
from torchvision import models
from dataset import Eye_img
from torch.utils.data import DataLoader
from visualize import Visualizer
from torchnet import meter
from cnn_finetune import make_model
import argparse
from torch import nn
from torchvision import models
from model import My_gcn
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pretrainedmodels
from zoom import zin

class My_loss(nn.Module):
    def __init__(self):
        super(My_loss,self).__init__()
        self.loss_fun = nn.CrossEntropyLoss()
    
    def forward(self, pre_label, label):
        loss1 = self.loss_fun(pre_label,label)
        losscut = t.tensor([0.8]).type(type(loss1))
        if loss1 < 0.8:
                loss1.data = losscut.data
        kappa = self.continuous_kappa(pre_label.max(dim=1)[1],label)
        return 0.5*loss1 - kappa

    def one_hot(self,vec, m=None):
        if m is None:
            m = int(np.max(vec)) + 1
        return np.eye(m)[vec].astype('int32')

    def continuous_kappa(self, y, t, y_pow=1, eps=1e-15):
        y, t = np.array(y), np.array(t)
        if y.ndim == 1:
            y = self.one_hot(y, m=5)

        if t.ndim == 1:
            t = self.one_hot(t, m=5)

        # Weights.
        num_scored_items, num_ratings = y.shape
        ratings_mat = np.tile(np.arange(0, num_ratings)[:, None],
                                reps=(1, num_ratings))
        ratings_squared = (ratings_mat - ratings_mat.T) ** 2
        weights = ratings_squared / float(num_ratings - 1) ** 2

        if y_pow != 1:
            y_ = y ** y_pow
            y_norm = y_ / (eps + y_.sum(axis=1)[:, None])
            y = y_norm

        hist_rater_a = np.sum(y, axis=0)
        hist_rater_b = np.sum(t, axis=0)

        conf_mat = np.dot(y.T, t)

        nom = weights * conf_mat
        denom = (weights * np.dot(hist_rater_a[:, None],
                                hist_rater_b[None, :]) /
                num_scored_items)
        return 1 - nom.sum() / denom.sum()#, conf_mat, hist_rater_a, hist_rater_b, nom, denom

      
def kappa(testData, k): 
    testData = np.array(testData)
    dataMat = np.mat(testData)
    P0_ = 0.0
    Pe_ = 0.0
#     for i in range(k):
#         P0_ += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1).reshape(-1,1)
    ysum = np.sum(dataMat, axis=0).reshape(-1,1)
    for i in range(k):
        for j in range(k):
            Pe_ += pow((i-j)*1.0/(k-1),2) * xsum[i] * ysum[j]
            P0_ += pow((i-j)*1.0/(k-1),2) * dataMat[i,j]
    P0_ = P0_/dataMat.sum()
    Pe_ = Pe_/xsum.dot(ysum.T).sum()
    cohens_coefficient = 1-P0_/Pe_
#     xsum = np.sum(dataMat, axis=1)
#     ysum = np.sum(dataMat, axis=0)
#     #xsum是个k行1列的向量，ysum是个1行k列的向量
#     Pe  = float(ysum*xsum)/testData.sum()**2
#     P0_ = float(P0_/testData.sum()*1.0)
#     cohens_coefficient = float((P0_-Pe)/(1-Pe))
    return t.tensor(cohens_coefficient).squeeze()

def train(opt):
    vis = Visualizer(opt.visname)

    # model = make_model('inceptionresnetv2',num_classes=5,pretrained=True,input_size=(512,512),dropout_p=0.6)
    model = zin()
    # model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model = model.cuda()

    loss_fun = t.nn.CrossEntropyLoss()
    # loss_fun = My_loss()
    lr = opt.lr
    # optim = t.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    optim = t.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-5,momentum=0.9)

    loss_meter = meter.AverageValueMeter()
    con_matx = meter.ConfusionMeter(5)
#     con_matx_temp = meter.ConfusionMeter(5)
    pre_loss = 1e10
    pre_acc = 0

    for epoch in range(100):
        loss_meter.reset()
        con_matx.reset()
        train_data = Eye_img(train=True)
        val_data = Eye_img(train=False)
        train_loader = DataLoader(train_data, opt.batchsize, True)
        val_loader = DataLoader(val_data, opt.batchsize, False)
        
        for ii,(imgs, data, label) in enumerate(train_loader):
            print('epoch:{}/{}'.format(epoch, ii))
        #     con_matx_temp.reset()
            # print(data.size())
            imgs = imgs.cuda()
            data = data.cuda()
            label = label.cuda()

            optim.zero_grad()
            pre_label = model(imgs, data)

        #     con_matx_temp.add(pre_label.detach(), label.detach())

        #     temp_value = con_matx_temp.value()
        #     temp_kappa = kappa(temp_value,5)
            loss = loss_fun(pre_label, label)
            loss.backward()
            optim.step()

            loss_meter.add(loss.item())
            con_matx.add(pre_label.detach(), label.detach())

            if (ii+1)%opt.printloss == 0:
                vis.plot('loss', loss_meter.value()[0])
        val_cm, acc, kappa_ = val(model, val_loader)
        if pre_acc < acc:
            pre_acc = acc
            t.save(model, 'inres_{}.pth'.format(epoch))
        vis.plot('acc', acc)
        vis.plot('kappa', kappa_)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                    epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), 
                    train_cm=str(con_matx.value()),lr=lr))
        
        # if loss_meter.value()[0] > pre_loss:
        #     lr = lr*0.7
        #     for param_group in optim.param_groups:
        #         param_group['lr'] = lr
        # pre_loss = loss_meter.value()[0]

@t.no_grad()
def val(model, val_loader):
    model.eval()
    con_matx = meter.ConfusionMeter(5)

    for ii ,(data, label) in enumerate(val_loader):
        print(ii)
        data = data.cuda()
        label = label.cuda()

        pre_ = model(data)

        con_matx.add(pre_.detach(), label.detach())
    cm_value = con_matx.value()
    kappa_ = kappa(cm_value,5)
    model.train()
    cm_sum = 0
#     kap_sum = [0,0,0]
    for i in range(5):
        # kap_sum[0] = 0
        # kap_sum[1] = 0
        cm_sum += cm_value[i][i]
        # for j in range(5):
        #     kap_sum[0] += cm_value[i][j]
        #     kap_sum[1] += cm_value[j][i]
        # kap_sum[2] += kap_sum[0]*kap_sum[1]
    acc = 100.*(cm_sum)/cm_value.sum()
#     Pe = kap_sum[2]*1.0/pow(cm_value.sum(),2)
#     kappa = (acc/100.-Pe)/(1-Pe)*100.
    return con_matx, acc, kappa_

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--batchsize", type=int, default=2,
            help="batch_size")
    parser.add_argument("--printloss", type=int, default=1,
            help="print_loss")
    parser.add_argument("--visname", type=str, default='sdc',
            help="visname")
    parser.add_argument("--lr", type=float, default=0.001,
            help="lr")
    args = parser.parse_args()
    print(args)
    train(args)
