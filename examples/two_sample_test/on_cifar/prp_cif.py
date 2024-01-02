import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
# from utils_HD import MatConvert, Pdist2, MMDu, TST_MMD_adaptive_bandwidth, TST_MMD_u, TST_ME, TST_SCF, TST_C2ST_D, TST_LCE_D
import ot
from ot_mmd.mmdot import solve_apgd
from ot_mmd.utils import get_dist, createLogHandler

# Setup seeds
os.makedirs("images", exist_ok=True)
np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True
is_cuda = True

# parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--ldas", nargs="+", type=float, default=100)
parser.add_argument("--ohps", nargs="+", type=float, default=-1.0)
parser.add_argument('--case', type=str, default='unb')
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--ktype', type=str, default='rbf')
parser.add_argument('--method', type=str, default='mmdot')
parser.add_argument('--only_validation', type=int, default=0)
parser.add_argument('--crit', default=None)
parser.add_argument('--log_msg', default="cif")
parser.add_argument('--p', type=int, default=2)

parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate for C2STs")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n", type=int, default=1000, help="number of samples in one set")
opt = parser.parse_args()

logger = createLogHandler(f"{opt.log_msg}.csv", str(os.getpid()))
dtype = torch.float
device = torch.device("cuda:0")
cuda = True if torch.cuda.is_available() else False
N_per = 100 # permutation times
alpha = 0.05 # test threshold
N1 = opt.n # number of samples in one set
K = 10 # number of trails
N = 100 # number of test sets
N_f = 100.0 # number of test sets (float)

list_lda = opt.ldas
list_ohp = opt.ohps
only_validation = opt.only_validation
ktype = opt.ktype
case = opt.case
max_iter = opt.max_iter
method = opt.method
crit = opt.crit

if isinstance(list_lda, float) or isinstance(list_lda, int):
    list_lda = [list_lda]
else:
    list_lda = list(list_lda)

if isinstance(list_ohp, float) or isinstance(list_ohp, int):
    list_ohp = [list_ohp]
else:
    list_ohp = list(list_ohp)

list_ohp = list_ohp + [None]  # for median heuristic

def eye_like(G):
    return torch.eye(*G.shape,out=torch.empty_like(G))

def get_rbf_G(khp=None, x=None, y=None, ridge=1e-10):
    """
    # NOTE: if dist is not None, it should be cost matrix**2. 
    If it is None, the function automatically computes euclidean**2.
    """
    if khp == None or khp == -1:  # take median heuristic
        khp = 0.5*torch.median(get_dist(x, y, p=1).view(-1))

    dist = get_dist(x, y)
    G = torch.exp(-dist/khp**2)
    if G.shape[0] == G.shape[1]:
        G = (G + G.T)/2
    G = G + ridge*eye_like(G)
    return G


def validate(S_v, list_lda, list_ohp):
    num_correct = {}  # no. of correct by each hp
    nor_margin = {}  # the normalized margin by each hp
    
    best = {"lda": 1, "ohp": -1, "nc": None, "nm": None}  # default lambda as 1, sigma for median heuristic
    
    for ohp in list_ohp:
        # initializing
        num_correct[ohp] = {}
        nor_margin[ohp] = {}
        for lda in list_lda:
            num_correct[ohp][lda] = 0
            nor_margin[ohp][lda] = 0

        for lda in list_lda:
            hp = {"lda": lda, "ohp": ohp}
            h, thr, val = run_test(S_v, hp)
            
            nor_margin[ohp][lda] += (val-thr)/(val + 1e-15)

            if h == 1:
                num_correct[ohp][lda] += 1
                
                # update best hp if ...
                if (best["nc"] is None or best["nc"] < num_correct[ohp][lda]) or (best["nc"] == num_correct[ohp][lda] and best["nm"] < nor_margin[ohp][lda]):  # (1st hp or this hp gets max correct) or (the nor_margin is max in case of a tie in num_correct)
                    best["lda"] = lda
                    best["ohp"] = ohp
                    best["nc"] = num_correct[ohp][lda]
                    best["nm"] = nor_margin[ohp][lda]
                    
    best_hp = {"lda": best["lda"], "ohp": best["ohp"]}
    return best_hp

def get_val(s1_t, s2_t, hp, G=None, C=None, indx=None, indy=None, call1=0, p=1):
    # returns obj value for a given hp
    lda = hp["lda"]
    ohp = hp["ohp"]
    
    if G is None:
        data_cat = torch.vstack([s1_t, s2_t])
        G = get_rbf_G(khp=ohp, x=data_cat, y=data_cat)
        indx = np.arange(s1_t.shape[0])
        indy = np.arange(s1_t.shape[0], data_cat.shape[0])
        C = get_dist(data_cat, data_cat, p=p)
        C = C/C.max()
    
    G1 = G[np.ix_(indx, indx)]
    G2 = G[np.ix_(indy, indy)]
    
    C_per = C[np.ix_(indx, indy)]
    
    a = torch.from_numpy(ot.unif(s1_t.shape[0])).to(dtype).to(device)
    b = torch.from_numpy(ot.unif(s2_t.shape[0])).to(dtype).to(device)
    v = {1: a, 2: b}

    _, obj_itr = solve_apgd(C_per, {1: G1, 2: G2}, v, max_iter, lda, crit=crit, tol=1e-6)

    val = obj_itr[-1].item()
    if call1:
        return val, G, C
    return val

def run_test(S_v, hp, p=opt.p):
    s1 = S_v[:N1, :]
    s2 = S_v[N1:, :]
    
    orig_value, G, C = get_val(s1, s2, hp, call1=1, p=p)
    perm_vals = []
    nxy = S_v.shape[0]
    for r in range(N_per):
        ind = np.random.choice(nxy, nxy, replace=False)
        indx, indy = ind[:N1], ind[N1:]
        
        perm_vals.append(get_val(S_v[indx], S_v[indy], hp, G, C, indx, indy, p=p))
        
    perm_vals = np.sort(perm_vals)
    threshold = perm_vals[np.int32(np.ceil(N_per * (1 - alpha)))]
    h = 1 if orig_value > threshold else 0
    return h, threshold, orig_value

# Configure data loader
dataset_test = datasets.CIFAR10(root='/home/saketh/data/cifar10', download=True,train=False,
                           transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000,
                                             shuffle=True, num_workers=1)
# Obtain CIFAR10 images
for i, (imgs, Labels) in enumerate(dataloader_test):
    data_all = imgs
    label_all = Labels
Ind_all = np.arange(len(data_all))
data_all = data_all.to(device)

# Obtain CIFAR10.1 images
data_new = np.load('/home/saketh/data/cifar10.1_v4_data.npy')
data_T = np.transpose(data_new, [0,3,1,2])
ind_M = np.random.choice(len(data_T), len(data_T), replace=False)
data_T = data_T[ind_M]
TT = transforms.Compose([transforms.Resize(opt.img_size),transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans = transforms.ToPILImage()
data_trans = torch.zeros([len(data_T),3,opt.img_size,opt.img_size], device=device)
data_T_tensor = torch.from_numpy(data_T).to(device)
for i in range(len(data_T)):
    d0 = trans(data_T_tensor[i])
    data_trans[i] = TT(d0)
Ind_v4_all = np.arange(len(data_T))

score_trial = []
ntrials = 10

# Repeat experiments K times (K = 10) and report average test power (rejection rate)
for kk in range(K):
    res = 0
    
    torch.manual_seed(kk * 19 + N1)
    torch.cuda.manual_seed(kk * 19 + N1)
    np.random.seed(seed=1102 * (kk + 10) + N1)

    # Collect CIFAR10 images
    Ind_tr = np.random.choice(len(data_all), N1, replace=False)
    Ind_te = np.delete(Ind_all, Ind_tr)

    # Collect CIFAR10.1 images
    np.random.seed(seed=819 * (kk + 9) + N1)
    Ind_tr_v4 = np.random.choice(len(data_T), N1, replace=False)
    Ind_te_v4 = np.delete(Ind_v4_all, Ind_tr_v4)
    New_CIFAR_tr = data_trans[Ind_tr_v4]
    New_CIFAR_te = data_trans[Ind_te_v4]

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)

    # Run two-sample test on the training set
    # Fetch training data
    s1 = data_all[Ind_tr]
    s2 = data_trans[Ind_tr_v4]
    S = torch.cat([s1, s2], 0)
    Sv = S.view(2 * N1, -1)
    best_hp = validate(Sv, list_lda, list_ohp)
    logger.info(f"{opt.n}, {kk}, {best_hp['lda']}, {best_hp['ohp']}")
    np.random.seed(1102)

    for k in range(N):
        # Fetch test data
        np.random.seed(seed=1102 * (k + 1) + N1)
        data_all_te = data_all[Ind_te]
        N_te = len(data_trans)-N1
        Ind_N_te = np.random.choice(len(Ind_te), N_te, replace=False)
        s1 = data_all_te[Ind_N_te]
        s2 = data_trans[Ind_te_v4]
        S = torch.cat([s1, s2], 0)
        Sv = S.view(2 * N_te, -1)
        
        h, thr, val = run_test(Sv, best_hp)
        
        res += h
    score_trial.append(res)
    logger.info(f"--------trial-score {score_trial[-1]}")
logger.info({"n":opt.n, "mean across trials": np.sum(score_trial)/(ntrials*N)})