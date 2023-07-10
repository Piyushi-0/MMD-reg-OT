import torch
from torch import sqrt
from torch.linalg import norm
from ot_mmd.utils import get_marginals, get_mmdsq_reg, proj_simplex
import numpy as np


def get_obj(C, G, lda, v, alpha, same_supp=1):
    alpha1, alphaT1 = get_marginals(alpha)
    reg_1, reg_2 = get_mmdsq_reg(alpha1, alphaT1, v, G, same_supp)
    E_c = torch.sum(alpha * C, dim=(1, 2))
    obj = E_c + lda*(reg_1 + reg_2)
    return obj

def get_grd(C, G, lda, v, alpha, same_supp):
    alpha1, alphaT1 = get_marginals(alpha)
    if same_supp:
        grd_1 = torch.matmul(G[1], (alpha1-v[1]).unsqueeze(-1))
        grd_2 = torch.matmul(G[2], (alphaT1-v[2]).unsqueeze(-1)).permute(0, 2, 1)
    else:
        raise NotImplementedError
    grd = C + 2*lda*(grd_1 + grd_2)
    return grd

def solve_apgd(C, G, v, max_itr, lda, crit=None, tol=None, same_supp=1, case="unb", verbose=0):
    if crit is not None:
        assert NotImplementedError  # TODO:
    b = C.shape[0]
    m, n = C[0].shape
    
    y = torch.ones_like(C)/(m*n)
    x_old = y
    
    t = 1
    G1_sqnorms = torch.norm(G[1], dim=(1, 2))**2
    G1_sums = torch.sum(G[1], dim=(1, 2))
    
    G2_sqnorms = torch.norm(G[2], dim=(1, 2))**2
    G2_sums = torch.sum(G[2], dim=(1, 2))
    
    ss = 1/(2*lda*(sqrt(n**2*G1_sqnorms + m**2*G2_sqnorms + 2*(G2_sums*G1_sums))))
    
    ss = ss.unsqueeze(-1).unsqueeze(-1)
    obj_init = get_obj(C, G, lda, v, y, same_supp)

    for itr in range(max_itr):
        grd = get_grd(C, G, lda, v, y, same_supp)
        if case =="unb":
            x_i = torch.clamp(y-ss*grd, min=0)
        else:
            x_i = proj_simplex(y-ss*grd)
        t_new = (1+np.sqrt(1+4*t**2))/2
        y = x_i + (t-1)*(x_i-x_old)/t_new
        x_old = x_i.clone()
        t = t_new
    obj_final = get_obj(C, G, lda, v, x_i, same_supp)
    assert torch.all(obj_init > obj_final), "No optimization! Obj_final={} Obj_initial={}".format(obj_final, obj_init)
    return x_i
