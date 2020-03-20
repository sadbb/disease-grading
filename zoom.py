import torch as t 
from torch import nn
from cnn_finetune import make_model


class zin(nn.Module):
    def __init__(self):
        super(zin,self).__init__()
        # model1 = make_model('inceptionresnetv2',num_classes=5,pretrained=True,input_size=(512,512),dropout_p=0.6)
        self.m_net1 = nn.Sequential()
        self.m_net2 = nn.Sequential()
        model1 = t.load('inres_80.pth').module
        self.m_net1.add_module('_features', model1._features)
        self.m_net1.add_module('dropout', model1.dropout)
        self.m_net2.add_module('pool', model1.pool)
        self.m_net2.add_module('_classifier', model1._classifier)
        for para in self.m_net1.parameters():
            para.requires_grad = False
        for para in self.m_net2.parameters():
            para.requires_grad = False
        self.a_net1 = nn.Sequential(
            nn.Conv2d(1536, 5, 1, 1),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True)
        )
        self.a_net2 = nn.Sequential(
            nn.Conv2d(1536,1536,3,1,padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            nn.Conv2d(1536,1536,3,1,padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),   
            nn.Conv2d(1536,5,1,1),
            Spatial_softmax()
        )
        modelc = make_model('inception_v3',num_classes=5,pretrained=True,input_size=(384,384),dropout_p=0.6)
        self.c_net1 = nn.Sequential()
        self.c_net1.add_module('_features', modelc._features)
        self.c_net1.add_module('dropout', modelc.dropout)
        self.c_net1.add_module('pool', modelc.pool)
        self.c_net2 = nn.Linear(2048+1536, 5)


    def forward(self, img, x):
        batch = x.size(0)
        h = self.m_net1(x)
        dm = self.m_net2.pool(h).squeeze() 
        ym = self.m_net2._classifier(dm)
        S = self.a_net1(h)
        A = self.a_net2(h)
        G = S*A # b*5*14*14
        ya = G.view(batch, 5, 14*14).sum(2)
        patch_id = getpatchs(G)
        # print(patch_id)
        # exit()
        flag = True
        patch = 0
        for i in range(len(patch_id)):
            for j in patch_id[i]:
                if flag:
                    patch = img[i, :, j[1]:j[1]+384, j[2]:j[2]+384].unsqueeze(0)
                    flag = False
                else:
                    patch = t.cat((patch, img[i, :, j[1]:j[1]+384, j[2]:j[2]+384].unsqueeze(0)),dim=0)
        p = self.c_net1(patch).squeeze()
        p = p.view(batch, 4, 2048).permute(0,2,1)
        p = p.max(dim=2)[0]
        yc = self.c_net2(t.cat((p, dm),dim=1))
        # ya = G.view(batch, 5, 14*14).sum(2)
        return nn.Softmax(dim=1)(ya + ym + yc)

        
class Spatial_softmax(nn.Module):
    def __init__(self):
        super(Spatial_softmax, self).__init__()
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self,x):
        # x B*C*k*k
        batch, c, w, _ = x.size()
        h = x.view(batch, c, w*w)
        h = self.softmax(h)
        return h.view(batch, c, w, w)

def getpatchs(data):
    batch = data.size(0)
    top_id = [[] for _ in range(batch)]
    # upsample = nn.Upsample(scale_factor=89, mode='bilinear')
    # data = upsample(data).view(batch, -1) # b*5*1246*1246
    data = nn.functional.interpolate(input=data, scale_factor=89, mode='bilinear')
    # for i in range(4):
    #     print(data[0,0,0,0:10])
    #     data[0,0,0,0:10] = 10
    for _ in range(4):
        # temp_data = nn.functional.interpolate(input=data, scale_factor=89, mode='bilinear')
        temp_data = data.view(batch, -1)
        # temp_data = upsample(data).view(batch, -1) # b*5*1246*1246
        max_id = temp_data.max(dim=1)[1]
        channel = max_id // (1246*1246)
        max_id -= channel*1246*1246
        x_id = max_id // 1246
        y_id = max_id - x_id * 1246
        # print(channel, x_id, y_id)
        # exit()
        patch_x = t.cat((x_id.unsqueeze(1), y_id.unsqueeze(1)),dim=1) - 192 
        patch_y = t.cat((x_id.unsqueeze(1), y_id.unsqueeze(1)),dim=1) + 192 
        patch_x = patch_x.clamp(0, 1245)
        patch_y = patch_y.clamp(0, 1245)
        for i in range(patch_x.size(0)):
            for j in range(patch_x.size(1)):
                patch_y[i][j] = min(1245, max(patch_x[i][j]+384, patch_y[i][j]))
                patch_x[i][j] = max(0, min(patch_y[i][j]-384, patch_x[i][j]))
        for i in range(patch_x.size(0)):
            top_id[i].append([channel[i], patch_x[i][0], patch_x[i][1]])
        # print(channel[0], x_id[0], y_id[0])
        # print(data[0, channel[0], x_id[0], y_id[0]])
        # print(data[0, :, patch_x[0][0]+192, patch_x[0][1]+192])
        for k in range(batch):
            data[k, :, patch_x[k][0]:patch_x[k][0]+384, patch_x[k][1]:patch_x[k][1]+384] = -1
        # print(data[0, :, patch_x[0][0]+192, patch_x[0][1]+192])
        # print('=============================')
    # print(top_id)
    # exit()
    return top_id
        
        
        




    
