import logging, ot, torch
import numpy as np
from sklearn import preprocessing
from ot.utils import proj_simplex as pot_proj_simplex


def set_seed(env, SEED=0):
    if SEED is None:
        return
    import random
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)


def get_t(arr, normalize=0, device=torch.device("cuda"), dtype=torch.float64, norm='l1'):
    if normalize:
        if len(arr.shape) > 2:
            b = arr.shape[0]
            for i in range(b):
                arr.append(preprocessing.normalize(arr[i], norm=norm))
            return torch.Tensor(arr, device=device, dtype=dtype)            
        arr = preprocessing.normalize(arr, norm=norm)
    return torch.from_numpy(arr).to(dtype).to(device)


def test_conv(obj_itr, tol=1e-3):
    cur_obj = obj_itr[-1]
    prv_obj = obj_itr[-2]
    rel_dec = abs(prv_obj-cur_obj)/(abs(prv_obj)+1e-10)
    if rel_dec < tol:
        return 1
    return 0    


def createLogHandler(log_file, job_name="_"):
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(log_file, mode='a')
    handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s; , %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_nrm_rgrad(x, grd):
    xegrad = x*grd
    lda = -torch.sum(xegrad, dim=[-1, -2])
    if x.dim() == 3:
        lda = lda.unsqueeze(-1).unsqueeze(-1)
    rgrad = xegrad + lda*x
    nrm_rgrad = torch.sum(torch.nan_to_num(rgrad**2/x), dim=[-1, -2])
    return nrm_rgrad


def sq_mnorm(vec, G):
    if G.dim() == 2:
        return torch.dot(torch.matmul(G, vec), vec)
    
    vec = vec.unsqueeze(-1)
    Gv = torch.matmul(G, vec)
    return torch.einsum("bmo,bmo->b", Gv, vec)


def get_marginals(b_alpha):
    b_alpha1 = torch.sum(b_alpha, axis=-1)
    b_alphaT1 = torch.sum(b_alpha, axis=-2)
    return b_alpha1, b_alphaT1


def get_mmdsq_reg(alpha1, alphaT1, v, G, same_supp):
    if same_supp:
        reg_1 = sq_mnorm(alpha1-v[1], G[1])
        reg_2 = sq_mnorm(alphaT1-v[2], G[2])
    else:
        if G.dim() == 3:  # TODO: vectorized version for this.
            raise NotImplemented    
        m = v[1].shape[0]
        G1 = G[:m, :m]
        G_1 = G[:, :m]
        G2 = G[m:, m:]
        G_2 = G[:, m:]
        reg_1 = sq_mnorm(alpha1, G) + sq_mnorm(v[1], G1) - 2*alpha1.dot(torch.mv(G_1, v[1]))
        reg_2 = sq_mnorm(alphaT1, G) + sq_mnorm(v[2], G2) - 2*alphaT1.dot(torch.mv(G_2, v[2]))
    return reg_1, reg_2


def eye_like(G):
    if(len(G.shape) == 3):
        return torch.eye(*G.shape[-2:], out=torch.empty_like(G)).repeat(G.shape[0], 1, 1)
    else: 
        return torch.eye(*G.shape,out=torch.empty_like(G))


def get_dist(x, y, p=2, dtype="euc", khp=None):
    x = x.unsqueeze(1) if x.dim() == 1 else x
    y = y.unsqueeze(1) if y.dim() == 1 else y

    C = torch.cdist(x, y)

    if p == 2 or "ker" in dtype:
        C = C**2
        if "rbf" in dtype:
            C = 2-2*get_G(dist=C, ktype="rbf", khp=khp, x=x, y=y)
        if "imq" in dtype:
            C = 2/khp**(0.5)-2*get_G(dist=C, ktype="imq", khp=khp, x=x, y=y)
    if "ker" in dtype and p == 1:
        C = C**(0.5)
    return C


def get_G(dist=None, ktype="rbf", khp=None, x=None, y=None, ridge=1e-10):
    """
    # NOTE: if dist is not None, it should be cost matrix**2. 
    If it is None, the function automatically computes euclidean**2.
    """
    if ktype in ["rbf", "imq", "imq_v2"]:
        if khp == None or khp == -1:  # take median heuristic
            khp = 0.5*torch.median(get_dist(x, y).view(-1))
        if dist is None:
            dist = get_dist(x, y)
    if ktype == "lin":
        if x.dim() == 2:
            G = torch.einsum('md,nd->mn', x, y)
        else:
            G = torch.einsum('bmd,nd->bmn', x, y)
    elif ktype == "rbf":
        G = torch.exp(-dist/(2*khp))
    elif ktype == "imq":
        G = (khp + dist)**(-0.5)
    elif ktype == "imq_v2":
        G = ((1+dist)/khp)**(-0.5)

    if(len(G.shape)==2):
        if G.shape[0] == G.shape[1]:
            G = (G + G.T)/2
    elif(G.shape[1] == G.shape[2]):
        G = (G + G.permute(0, 2, 1))/2
    G = G + ridge*eye_like(G)
    return G

def get_cost_G(x, y, khp, ktype, p=2):
    # None means taking median-heuristic
    C = get_dist(x, y, p)
    C = C/C.max()
    
    G1 = get_G(x=x, y=x, khp=khp, ktype=ktype)
    G2 = get_G(x=y, y=y, khp=khp, ktype=ktype)
    G = {1: G1, 2: G2}
    return C, G

def proj_simplex(v):
    # TODO: vectorize algo for proj_simplex.
    if v.dim() == 3:
        b = v.shape[0]
        proj_vs = []
        for i in range(b):
            shape = v[i].shape
            proj_vs.extend(pot_proj_simplex(v[i].view(-1, 1)).view(shape))
        return proj_vs
    shape = v.shape
    return pot_proj_simplex(v.view(-1, 1)).view(shape)

def get_genw_tv(case, v, C, lda):
    def get_A(m1, m2):
        ix1 = np.arange(m1*m2)
        a1 = np.zeros((m1, m1*m2))
        for i in range(m1):
            a1[i, ix1[m2*i:m2*(i+1)]] = 1 # for sum(axis = 1)
        a2 = np.zeros((m2, m1*m2))
        for i in range(m2):
            a2[i, ix1[i:m1*m2+1:m2]] = 1 # for sum(axis = 0)
        A = np.vstack([a1, a2])
        return A
    from scipy.optimize import linprog
    m1 = v[1].shape[0]
    m2 = v[2].shape[0]

    A_ub = get_A(m1, m2)
    if case == "bal":
        bounds = (0, 1)
    else:
        bounds = (0, None)
    b_ub = np.hstack([v[1], v[2]])
    cost = C.reshape(-1)-2*lda*np.ones(m1*m2)
    res = linprog(c = cost, A_ub=A_ub, b_ub = b_ub, bounds = bounds)
    assert (not res.status) and (res.success)
    return res
