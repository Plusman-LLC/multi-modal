#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
 L.ConvolutionND(3... = nn.Conv3d
 L.DeconvolutionND(3... = nn.ConvTranspose3d
 F.concat([]) = torch.cat([],dim=1)
 F.average_pooling_nd(x) = nn.AvgPool3d(2,2)(x)
 F.sum(x,axis=1,keepdims=True) = torch.sum(x,dim=1,keepdim=True)
 chainer.config.enable_backprop = False -> with torch.no_grad():
"""

resize = lambda x,s: F.interpolate(x, size=s, mode='trilinear', align_corners=False)

class HalfUNET(nn.Module):
    def __init__(self, in_ch, n_seg):
        super(HalfUNET, self).__init__()
                #in_ch, out_ch, ksize, stride, pad
        self.c0 = nn.Conv3d(in_ch, 32, 3, 1, 1) # (512 - 3 + 2*1)/1 + 1 = 512
        self.c1 = nn.Conv3d(32, 64, 4, 2, 1)   # (512 - 4 + 2*1)/2 + 1 = 256
        self.c2 = nn.Conv3d(64, 64, 3, 1, 1)   # (256 - 3 + 2*1)/1 + 1 = 256
        self.c3 = nn.Conv3d(64, 128, 4, 2, 1)  # (256 - 4 + 2*1)/2 + 1 = 128
        self.c4 = nn.Conv3d(128, 128, 3, 1, 1) # (128 - 3 + 2*1)/1 + 1 = 128

        self.dc4 = nn.ConvTranspose3d(256, 128, 4, 2, 1) # 2*(128 - 1) + 4 - 2*1 = 256
        self.dc3 = nn.Conv3d(128, 64, 3, 1, 1)    # (256 - 3 + 2*1)/1 + 1 = 256
        self.dc2 = nn.ConvTranspose3d(128, 64, 4, 2, 1)  # 2*(256 - 1) + 4 - 2*1 = 512
        self.dc1 = nn.Conv3d(64, 32, 3, 1, 1)     # (512 - 3 + 2*1)/1 + 1 = 512
        self.dc0 = nn.Conv3d(64, n_seg, 3, 1, 1)  # (512 - 3 + 2*1)/1 + 1 = 512

        self.bnc0 = nn.GroupNorm(32,32)
        self.bnc1 = nn.GroupNorm(64,64)
        self.bnc2 = nn.GroupNorm(64,64)
        self.bnc3 = nn.GroupNorm(128,128)
        self.bnc4 = nn.GroupNorm(128,128)

        self.bnd4 = nn.GroupNorm(128,128)
        self.bnd3 = nn.GroupNorm(64,64)
        self.bnd2 = nn.GroupNorm(64,64)
        self.bnd1 = nn.GroupNorm(32,32)

    def forward(self,x):
        e0 = F.relu(self.bnc0(self.c0(x)))  #; print(e0.data.shape)
        e1 = F.relu(self.bnc1(self.c1(e0))) #; print(e1.data.shape)
        e2 = F.relu(self.bnc2(self.c2(e1))) #; print(e2.data.shape)
        e3 = F.relu(self.bnc3(self.c3(e2))) #; print(e3.data.shape)
        e4 = F.relu(self.bnc4(self.c4(e3))) #; print(e4.data.shape)

        d4 = F.relu(self.bnd4(self.dc4(torch.cat([e3, e4],dim=1)))) #; print(d4.data.shape)
        d3 = F.relu(self.bnd3(self.dc3(d4)))                 #; print(d3.data.shape)
        d2 = F.relu(self.bnd2(self.dc2(torch.cat([e2, resize(d3, e2.shape[2:])],dim=1)))) #; print(d2.data.shape)
        d1 = F.relu(self.bnd1(self.dc1(d2)))                 #; print(d1.data.shape)
        h = torch.cat([e0, resize(d1, e0.shape[2:])],dim=1)
        d0 = self.dc0(h)                    #; print(d0.data.shape)

        return d0, h

class HALF_UDEN(nn.Module):

    def __init__(self,in_ch=1, out_ch=1):
        n_seg = 30
        super(HALF_UDEN  , self).__init__()
        self.EC_cls=HalfUNET(in_ch,n_seg*out_ch)
        self.EC_msk=nn.Conv3d(in_ch, (n_seg-1)*out_ch, 3, 1, 1)
        self.n_seg = n_seg
        self.out_ch = out_ch

    def forward(self, x):
#        with torch.no_grad():
        cls = self.EC_cls(x)
        shape = cls.shape
        cls = torch.reshape(cls,(shape[0],self.out_ch,self.n_seg,shape[2],shape[3],shape[4]))
        cls = F.softmax(cls,dim=2)
        
        msk = self.EC_msk(x)
        msk = torch.reshape(msk,(shape[0],self.out_ch,self.n_seg-1,shape[2],shape[3],shape[4]))
        temp = msk[:,:,0:1]*0#torch.tensor(np.zeros((x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]),dtype=np.float32)).
        msk = torch.cat([temp,msk],dim=2)
        
#        self.cls, self.msk = cls, msk
        y = torch.sum(cls*msk,axis=2,keepdims=False)
        
        return y