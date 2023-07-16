import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


import pdb

def conv1x1(in_channels, out_channels,stride=1,padding=0,dilation=1,bias=False,relu=False):
    if bias and relu:
        out = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                            stride=stride, padding=padding, dilation=dilation,bias=bias),
            torch.nn.ReLU(inplace=True),
        )
    elif (not bias) and relu:
        out = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                            stride=stride, padding=padding, dilation=dilation,bias=bias),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(inplace=True)
        )
    elif (not bias) and (not relu):
        out = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                            stride=stride, padding=padding, dilation=dilation,bias=bias),
            torch.nn.BatchNorm2d(num_features=out_channels),
        )
    else:# bias and (not relu):
        out = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                            stride=stride, padding=padding, dilation=dilation,bias=bias),
        )
    return  out

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        dropout = 0.3
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class localTransformer(nn.Module):
    def __init__(self, in_channels, depth):
        super(localTransformer,self).__init__()
        down_channels = in_channels   
        self.in_channels = in_channels
        self.out_channels = down_channels
        self.iters = depth

        if in_channels == 32 or in_channels == 64:
            self.dim_head = 4
        elif in_channels == 128:
            self.dim_head = 8
        else:
            self.dim_head = 16

        head_dim = in_channels // self.dim_head
        self.scale = head_dim ** -0.5
        self.hidden_dim = head_dim

        self.conv1x1_up = torch.nn.Conv2d(in_channels=down_channels, out_channels=self.in_channels ,kernel_size=1)
        self.conv1x1_down = torch.nn.Conv2d(in_channels=self.in_channels*2, out_channels=self.in_channels ,kernel_size=1)
 
        self.get_k = conv1x1(in_channels=in_channels,out_channels=down_channels ,bias=False, relu=False)
        self.get_q = conv1x1(in_channels=in_channels,out_channels=down_channels ,bias=False, relu=False)
        self.get_v = conv1x1(in_channels=in_channels, out_channels=down_channels,bias=True, relu=False)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.mlp_head = MLP(self.dim_head, self.hidden_dim)


    def qkv(self,x):
        q = self.get_q(x)
        k = self.get_k(x)
        v = self.get_v(x)

        return q, k, v

    def reshape_values(self,x,kernel_size,padding,dilation):
        B,C,H,W = x.shape
        self.H = H
        self.W = W
        x = torch.nn.functional.unfold(x,kernel_size=kernel_size,padding=padding,dilation=dilation)
        x = x.view(B,C,kernel_size*kernel_size,H*W).view(B,self.dim_head,-1,kernel_size*kernel_size, H*W)
        return x


    def attention(self, k, q ,v):
        B, g, C, K, L = k.shape
        B, g, C, M, L = q.shape
        self.B = B
        self.L = L

        attention = torch.einsum('bgcil,bgcml->bgiml',k,q) * self.scale #shape of [B,g,kernel*kernel,1,L]
        attention = torch.nn.functional.softmax(attention,dim=2)

        out = torch.einsum('bgiml,bgcil->bgcml',attention,v)
        # out = out.reshape(B,-1,1,L).view(B,-1,self.H,self.W)

        return out

    def forward(self,x):
        x0 = x
        for i in range(self.iters):        
            q, k ,v = self.qkv(x)

            q = self.reshape_values(q, kernel_size=1,padding=0,dilation=1)
            k = self.reshape_values(k, kernel_size=1,padding=0,dilation=1)

            v = self.reshape_values(v,kernel_size=1,padding=0,dilation=1)
            v = self.attention(k,q,v).transpose(1,4)
            # pdb.set_trace()
            v = self.mlp_head(v).transpose(1,4)
            v = v.reshape(self.B,-1,1,self.L).view(self.B,-1,self.H,self.W)     #resize shape
            
            x = v
        
        out = torch.cat((x0,x),dim=1)
        x = self.conv1x1_down(out)

        return x














