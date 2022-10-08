import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
import torch_scatter
import numpy


class Trans_dim(nn.Module):
    def __init__(self, dim_in, dim_out, use_bn=False) -> None:
        super(Trans_dim, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out),
            nn.Conv2d(dim_out, dim_out, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out)
        )
    def forward(self, cylinder_polar):

        return self.conv(cylinder_polar)

class Cross_Attn(nn.Module):
    def __init__(self, dim_cylinder, dim_polar):
        self.dim_cylinder = dim_cylinder
        super(Cross_Attn, self).__init__()
        self.q = nn.Linear(dim_polar, dim_cylinder)
        self.kv = nn.Linear(dim_cylinder, dim_cylinder*2)
        
    def forward(self, cylinder_fea, polar_fea):
        current_device = polar_fea.get_device()
        layer_polar = polar_fea  # [B, dim, x, y]
        layer_cylinder = cylinder_fea  # [B, dim, x, y, z]
        cylinder_features, cylinder_indices = layer_cylinder.features, layer_cylinder.indices
        unq, unq_inv, unq_cnt = \
            torch.unique(cylinder_indices[:,:3], return_inverse=True, return_counts=True, dim=0)
        cylinder_features = torch_scatter.scatter_max(cylinder_features,unq_inv, dim=0)[0]
        #   TODO
        # select the pillar with more voxels
        select_num = 500 if unq.shape[0] > 500 else unq.shape[0] // 2
        select_choice = torch.multinomial(unq_cnt.to(torch.float32), select_num)
        unq_select = unq[select_choice,:].type(torch.int64)
        cylinder_features = cylinder_features[select_choice]
        polar_features = polar_fea[unq_select[:,0],:,unq_select[:,1],unq_select[:,2]]

        q = self.q(polar_features)
        kv = self.kv(cylinder_features)
        k = kv[:,:self.dim_cylinder]
        v = kv[:,self.dim_cylinder:]
        out_put = (q@k.T)@v
        return cylinder_features, out_put

        # assert cylinder_polar.shape == layer_polar.shape
        # return cylinder_polar, layer_polar
        


class KD_Part(nn.Module):
    def __init__(self, dim_cylinder_list, dim_polar_list) -> None:
        super(KD_Part, self).__init__()
        self.cylinder_dim = dim_cylinder_list
        self.polar_dim = dim_polar_list
        self.trans_list = nn.ModuleList()
        for i in range(len(dim_cylinder_list)):
            self.trans_list.append(Cross_Attn(dim_cylinder_list[i], dim_polar_list[i]))

    def forward(self, cylinder_list, polar_list):
        cylinder_kd = []
        polar_kd = []
        for i in range(len(cylinder_list)):
            cylinder_polar, layer_polar = self.trans_list[i](cylinder_list[i], polar_list[i])
            cylinder_kd.append(cylinder_polar)
            polar_kd.append(layer_polar)
        return cylinder_kd, polar_kd