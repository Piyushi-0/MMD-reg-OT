import torch
from torch import sqrt
from torch.linalg import norm
from ot_mmd.utils import test_conv, get_nrm_rgrad, get_marginals, get_mmdsq_reg, proj_simplex
import numpy as np


def get_obj(C, G, lda, v, alpha, same_supp=1):
    alpha1, alphaT1 = get_marginals(alpha)
    reg_1, reg_2 = get_mmdsq_reg(alpha1, alphaT1, v, G, same_supp)
    E_c = torch.tensordot(alpha, C)
    obj = E_c + lda*(reg_1+reg_2)
    return obj


def get_grd(C, G, lda, v, alpha, same_supp=1):
    alpha1, alphaT1 = get_marginals(alpha)
    if same_supp:
        grd_1 = torch.matmul(G[1], alpha1-v[1])[:, None]
        grd_2 = torch.matmul(G[2], alphaT1-v[2])
    else:
        m = v[1].shape[0]
        G_r, G_l =  G[:, m:], G[:, :m]
        grd_1 = (torch.matmul(G, alpha1) - torch.matmul(G_l, v[1]))[:, None]
        grd_2 = torch.matmul(G, alphaT1) - torch.matmul(G_r, v[2])
    grd = C+2*lda*(grd_1+grd_2)
    return grd


def solve_apgd(C, G, v, max_itr, lda, crit=None, tol=1e-3, same_supp=1, case="bal", verbose=0):
    """solve via accelerated projected gd

    Args:
    C (_array_like_): cost matrix between source and target.
    G (_array_like_): Gram matrix with samples from source.
    v (_vector_): source distribution over samples.
    max_itr (_int_): for APGD.
    lda (_float_): lambda regularization hyperparameter.
    crit (str, optional): stopping criteria.
    tol (_float_, optional): threshold for riemannian gradient based stopping criteria.
    same_supp (int, optional): If supports match or not. Defaults to 1.
    case (str, optional): balanced or unbalanced measure.
    verbose (boolean, optional): whether to display convergence information.

    Returns:
    x_i (FloatTensor): OT plan
    obj_itr (list): objective over iterations, returned if verbose is 1.
    """
    if case == "unb":
        assert crit != "rgrad", "Not yet implemented Riemmanian gradient based criteria for unbalanced"
    
    m, n = C.shape
    y = torch.ones_like(C)/(m*n)
    x_old = y

    t = 1
    ss = 1/(2*lda*(sqrt(n**2*norm(G[1])**2 + m**2
                            * norm(G[2])**2 + 2*(G[1].sum()*
                                                G[2].sum()))))
    obj_itr = []
    obj_init = get_obj(C, G, lda, v, y, same_supp)
    
    for itr in range(max_itr):
        # update
        grd = get_grd(C, G, lda, v, y, same_supp)
        if case =="unb":
            x_i = torch.clamp(y-ss*grd, min=0)
        else:
            x_i = proj_simplex(y-ss*grd)
        obj_itr.append(get_obj(C, G, lda, v, x_i, same_supp))
        # check for convergence
        if crit == "obj" and itr>1 and test_conv(obj_itr, tol):
            break
        elif crit == "rgrad":  # based on the norm of Riemannian gradient
            grd_xi = get_grd(C, G, lda, v, x_i, same_supp)
            if get_nrm_rgrad(x_i, grd_xi) < tol:
                break
        # update intermediate variables
        t_new = (1+np.sqrt(1+4*t**2))/2
        y = x_i + (t-1)*(x_i-x_old)/t_new
        x_old = x_i.clone()
        t = t_new
    if verbose and itr < max_itr-1:
        print(f"Converged in {itr+1} iterations.")
    obj_final = obj_itr[-1] if crit == "obj" else get_obj(C, G, lda, v, x_i, same_supp)
    assert obj_final < obj_init, "No optimization! Obj_final={} Obj_initial={}".format(obj_final, obj_init)
    return x_i, obj_itr
