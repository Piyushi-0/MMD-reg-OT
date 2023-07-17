import torch
from torch import sqrt
from torch.linalg import norm
from ot_mmd.utils import test_conv, get_nrm_rgrad, get_marginals, get_mmdsq_reg, proj_simplex
import numpy as np


def get_obj(C, G, lda, v, alpha, rho):    
    cost_part = rho[1]*torch.tensordot(C[1], alpha[1]) + rho[2]*torch.tensordot(C[2], alpha[2])
    
    alpha1_1, alpha1_T1 = get_marginals(alpha[1])
    alpha2_1, alpha2_T1 = get_marginals(alpha[2])

    avg_alphaT1 = rho[1]*alpha1_T1+rho[2]*alpha2_T1
    
    reg1_1, reg1_2 = get_mmdsq_reg(alpha1_1, alpha1_T1, {1: v[1], 2: avg_alphaT1}, {1: G[1], 2: G['all']}, same_supp=1)
    reg2_1, reg2_2 = get_mmdsq_reg(alpha2_1, alpha2_T1, {1: v[2], 2: avg_alphaT1}, {1: G[2], 2: G['all']}, same_supp=1)
    
    lda1_part = rho[1]*reg1_1 + rho[2]*reg2_1
    lda2_part = rho[1]*reg1_2 + rho[2]*reg2_2

    obj = cost_part + lda[1]*lda1_part + lda[2]*lda2_part
    return obj

def get_grd(C, G, lda, v, alpha, rho):
    # returns gradients wrt the two alpha variables.
    alpha1_1, alpha1_T1 = get_marginals(alpha[1])
    alpha2_1, alpha2_T1 = get_marginals(alpha[2])

    avg_alphaT1 = rho[1]*alpha1_T1+rho[2]*alpha2_T1

    grd_1 = 0
    if rho[1]>0:
        grd_1 = rho[1]*C[1] + 2*lda[1]*rho[1]*torch.mv(G[1], alpha1_1-v[1])[:, None] + \
                2*lda[2]*rho[1]*(1-rho[1])*torch.mv(G['all'], alpha1_T1-avg_alphaT1) + \
                2*lda[2]*rho[2]*(-rho[1])*torch.mv(G['all'], alpha2_T1-avg_alphaT1)
    
    grd_2 = 0
    if rho[2]>0:
        grd_2 = rho[2]*C[2] + 2*lda[1]*rho[2]*torch.mv(G[2], alpha2_1-v[2])[:, None] + \
                2*lda[2]*rho[2]*(1-rho[2])*torch.mv(G['all'], alpha2_T1-avg_alphaT1) + \
                2*lda[2]*rho[1]*(-rho[2])*torch.mv(G['all'], alpha1_T1-avg_alphaT1)
    
    return grd_1, grd_2


def solve_apgd(C, G, v, max_itr, lda, rho={1: 0.5, 2: 0.5}, crit=None, tol=1e-3, case="bal", verbose=0):
    """
    Args:
        a : source distribution.
        b : target distribution.
        C : dictionary of cost matrices such that C[1] is over source samples & union of source & target samples.
                                                  C[2] is over target samples & union of source & target samples.
        G : dictionary of Gram matrices such that G[1] is over source-source samples.
                                                  G[2] is over target-target samples.
                                                  G['all'] is over the union of samples.
        lda : dictionary such that lda[1], lda[2] are regularization coefficients.
        rho : dictionary such that rho[1], rho[2] are the coefficients.
        crit (str, optional): stopping criteria.
        tol (_float_, optional): threshold for riemannian gradient based stopping criteria.
        case (str, optional): balanced or unbalanced measure.
        verbose (boolean, optional): whether to display convergence information.

    Returns:
        barycenter distribution supported over the union of source & target samples.
    """
    m1, m2 = C[1].shape[0], C[2].shape[0]
    m = m1+m2
    y = {1: torch.ones_like(C[1])/(m1*m), 2: torch.ones_like(C[2])/(m2*m)}
    x_old = y
    
    t = 1
    eta_1 = lda[2]*(1-rho[1])
    eta_2 = lda[2]*(1-rho[2])
    ss = {1: 1/(2*rho[1]*(sqrt((lda[1]*m)**2*norm(G[1])**2 + (eta_1*m1)**2
                            * norm(G['all'])**2 + 2*lda[1]*eta_1*(G[1].sum()*
                                                G['all'].sum())))) if rho[1] else 0,
          \
          2: 1/(2*rho[2]*(sqrt((lda[1]*m)**2*norm(G[2])**2 + (eta_2*m2)**2
                            * norm(G['all'])**2 + 2*lda[1]*eta_2*(G[2].sum()*
                                                G['all'].sum())))) if rho[2] else 0}
    
    obj_itr = []
    obj_init = get_obj(C, G, lda, v, y, rho)
    opt1 = opt2 = max_itr
    for itr in range(max_itr):
        # update
        grd1, grd2 = get_grd(C, G, lda, v, y, rho)
        if not itr:
            x_i = {1: torch.clamp(y[1]-ss[1]*grd1, min=0) if case == "unb" else proj_simplex(y[1]-ss[1]*grd1),
                   2: torch.clamp(y[2]-ss[2]*grd2, min=0) if case == "unb" else proj_simplex(y[2]-ss[2]*grd2)}
        elif opt1 == max_itr or opt2 == max_itr:
            x_i[1] = torch.clamp(y[1]-ss[1]*grd1, min=0) if case == "unb" else proj_simplex(y[1]-ss[1]*grd1)
            x_i[2] = torch.clamp(y[2]-ss[2]*grd2, min=0) if case == "unb" else proj_simplex(y[2]-ss[2]*grd2)
        
        obj_itr.append(get_obj(C, G, lda, v, x_i, rho))
        # check for convergence
        if crit == "obj" and itr>1 and test_conv(obj_itr, tol):
            break
        elif crit == "rgrad":
            grd1_xi, grd2_xi = get_grd(C, G, lda, v, x_i, rho)
            if get_nrm_rgrad(x_i[1], grd1_xi) < tol:
                opt1 = itr
            if get_nrm_rgrad(x_i[2], grd2_xi) < tol:
                opt2 = itr
        # update intermediate variables
        t_new = (1+np.sqrt(1+4*t**2))/2
        y = {1: x_i[1] + (t-1)*(x_i[1]-x_old[1])/t_new,
             2: x_i[2] + (t-1)*(x_i[2]-x_old[2])/t_new}
        x_old = {1: x_i[1].clone(), 2: x_i[2].clone()}
        t = t_new
    if verbose and (opt1 < max_itr and opt2 < max_itr):
        print(f"Converged early.")
    obj_final = obj_itr[-1] if crit == "obj" else get_obj(C, G, lda, v, x_i, rho)
    assert obj_final < obj_init, "No optimization! Obj_final={} Obj_initial={}".format(obj_final, obj_init)
    
    bary = rho[1]*x_i[1].sum(axis=0) + rho[2]*x_i[2].sum(axis=0)
    return bary, obj_itr
