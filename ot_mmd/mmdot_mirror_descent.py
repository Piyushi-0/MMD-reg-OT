import torch
from torch import sqrt
from ot_mmd.utils import get_nrm_rgrad, get_marginals, get_mmdsq_reg, sq_mnorm


def get_obj(C, G, lda, v, alpha, same_supp=1, reg_type="mmd"):
    alpha1, alphaT1 = get_marginals(alpha)
    reg_1, reg_2 = get_mmdsq_reg(alpha1, alphaT1, v, G, same_supp)
    if reg_type == "mmd":
        reg_1 = sqrt(reg_1)
        reg_2 = sqrt(reg_2)
    E_c = torch.tensordot(alpha, C)
    obj = E_c + lda*(reg_1 + reg_2)
    return obj


def get_grd(C, G, lda, v, alpha, same_supp=1, reg_type="mmd"):
    alpha1, alphaT1 = get_marginals(alpha)
    if same_supp:
        vec1 = alpha1-v[1]
        vec2 = alphaT1-v[2]

        grd_1 = torch.matmul(G[1], vec1)[:, None]
        grd_2 = torch.matmul(G[2], vec2)
        
        if reg_type == "mmd":
            reg_1 = sqrt(sq_mnorm(vec1, G[1]))
            reg_2 = sqrt(sq_mnorm(vec2, G[2]))
            grd = C+lda*(grd_1/reg_1+grd_2/reg_2)
        else:
            grd = C+2*lda*(grd_1+grd_2)
    else:
        raise NotImplementedError    
    return grd

def update_vars_md(alpha, grd, case):
    s = 1/torch.norm(grd, p=torch.inf)
    alpha = alpha*torch.exp(-grd*s)
    if case == "bal":
        alpha = alpha/alpha.sum()
    return alpha

def solve_md(C, G, v, max_itr, lda, crit=None, tol=1e-4, same_supp=1, case="bal", reg_type="mmd", verbose=0):
    
    """solve via mirror descent

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
    reg_type (str, optional): mmd or mmd^2.
    verbose (boolean, optional): whether to display convergence information.

    Returns:
    x_i (FloatTensor): OT plan
    obj_itr (list): objective over iterations, returned if verbose is 1.
    """
    if case == "unb":
        assert crit != "rgrad", "Not yet implemented Riemmanian gradient based criteria for unbalanced"
    
    m, n = C.shape
    alpha = torch.ones_like(C)/(m*n)
    best_alpha = None
    best_obj = torch.inf
    
    obj_itr = []
    obj_init = get_obj(C, G, lda, v, alpha, same_supp, reg_type).item()
    grd = get_grd(C, G, lda, v, alpha, same_supp, reg_type)
    obj_itr.append(obj_init)
    
    for itr in range(max_itr):
        # update
        try:
            alpha = update_vars_md(alpha, grd, case)
        except Exception as e:
            print(e)  # should be the case of 0 grd

        obj_itr.append(get_obj(C, G, lda, v, alpha, same_supp, reg_type).item())
        
        if obj_itr[-1] < best_obj:
            best_obj = obj_itr[-1]
            best_alpha = alpha.clone()
        
        grd = get_grd(C, G, lda, v, alpha, same_supp, reg_type)
        if crit == "rgrad" and get_nrm_rgrad(alpha, grd) < tol:
            break
    if verbose and itr < max_itr-1:
        print(f"Converged in {itr+1} iterations.")
    best_obj = min(obj_itr)
    assert best_obj <= obj_init, "No optimization! Obj_final={} Obj_initial={}".format(best_obj, obj_init)
    return best_alpha, obj_itr
