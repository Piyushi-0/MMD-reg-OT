import torch
from ot_mmd.utils import get_marginals

    
def get_kl(v1, v2, case, eps=1e-10):
    v1 = v1 + eps
    v2 = v2 + eps
    kl = torch.sum(torch.where(v1 != 0, v1*torch.log(v1/v2), 0))
    if case == "unb":
        kl = kl-v1.sum() + v2.sum()
    return kl

def get_entropy(alpha, case, eps=1e-10):
    alpha = alpha + eps
    entropy = torch.sum(torch.where(alpha != 0, alpha * torch.log(alpha), 0))
    if case == "unb":
        entropy = entropy - alpha.sum()
    return entropy

def get_obj(alpha, bary, v, C, lda, coeff_entr, rho={1: 0.5, 2: 0.5}, case="bal"):
    cost_part = rho[1]*torch.tensordot(alpha[1], C[1]) + rho[2]*torch.tensordot(alpha[2], C[2])
    
    alpha1_1, alpha1_T1 = get_marginals(alpha[1])
    alpha2_1, alpha2_T1 = get_marginals(alpha[2])
    
    lda1_part = rho[1]*get_kl(alpha1_1, v[1], case) + rho[2]*get_kl(alpha2_1, v[2], case)
    lda2_part = rho[1]*get_kl(alpha1_T1, bary, case) + rho[2]*get_kl(alpha2_T1, bary, case)

    obj = cost_part + lda[1]*lda1_part + lda[2]*lda2_part
    obj += coeff_entr*(rho[1]*get_entropy(alpha[1], case)+rho[2]*get_entropy(alpha[2], case))
    return obj

def get_grd(alpha, bary, v, C, lda, coeff_entr, rho={1: 0.5, 2: 0.5}, case="bal"):
    eps = 1e-10
    
    alpha[1] = alpha[1] + eps
    alpha[2] = alpha[2] + eps
    bary = bary + eps
    
    alpha1_1, alpha1_T1 = get_marginals(alpha[1])
    alpha2_1, alpha2_T1 = get_marginals(alpha[2])
    
    grd_bary = -lda[2]*(rho[1]*alpha1_T1 + rho[2]*alpha2_T1)/bary
    grd_1 = grd_2 = 0
    if rho[1]>0:
        term1 = torch.log(alpha1_1)-torch.log(v[1])
        term2 = torch.log(alpha1_T1)-torch.log(bary)
        if case == "bal":
            term1 += 1
            term2 += 1
        grd_1 = rho[1]*(C[1] + lda[1]*term1[:, None] + lda[2]*term2)
    
    if rho[2]>0:
        term1 = torch.log(alpha2_1)-torch.log(v[2])
        term2 = torch.log(alpha2_T1)-torch.log(bary)
        if case == "bal":
            term1 += 1
            term2 += 1
        grd_2 = rho[2]*(C[2] + lda[1]*term1[:, None] + lda[2]*term2)

    grd_1 += rho[1]*coeff_entr*(1+torch.log(alpha[1])) if case == "bal" else rho[1]*coeff_entr*torch.log(alpha[1])
    grd_2 += rho[2]*coeff_entr*(1+torch.log(alpha[2])) if case == "bal" else rho[2]*coeff_entr*torch.log(alpha[2])
    
    return grd_1, grd_2, grd_bary


def solve_md(v, C, lda, max_itr, coeff_entr, rho={1: 0.5, 2: 0.5}, case="bal"):
    
    def update_vars(var, grd, case):
        s = 1/torch.norm(grd, torch.inf)
        var = var*torch.exp(-grd*s)
        if case == "bal":
            var = var/var.sum()
        return var
        
    alpha = {1: torch.ones_like(C[1])/C[1].numel(),
             2: torch.ones_like(C[2])/C[2].numel()}
    bary = (torch.ones(C[1].shape[1])/C[1].shape[1]).to(C[1].dtype).to(C[1].device)
    obj_itr = []
    bary_best = None
    best_itr = None
    
    for itr in range(max_itr):
        obj_itr.append(get_obj(alpha, bary, v, C, lda, coeff_entr, rho))
        if best_itr is None or obj_itr[best_itr] > obj_itr[-1]:
            best_itr = itr
            bary_best = bary.clone()
        grd_1, grd_2, grd_bary = get_grd(alpha, bary, v, C, lda, coeff_entr, rho)
        if rho[1] > 0:
            try:  # error triggered when optimality has been reached
                alpha[1] = update_vars(alpha[1], grd_1, case)
            except Exception as e:
                print(e)
                pass

        if rho[2] > 0:
            try:  # error triggered when optimality has been reached
                alpha[2] = update_vars(alpha[2], grd_2, case)
            except Exception as e:
                print(e)
                pass

        try:  # error triggered when optimality has been reached
            bary = update_vars(bary, grd_bary, case)
        except Exception as e:
            print(e)
            pass
    
    return bary_best, obj_itr
