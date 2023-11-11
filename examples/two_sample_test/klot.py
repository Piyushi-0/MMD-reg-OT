from ot.unbalanced import sinkhorn_unbalanced2 as klot
import argparse
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import ot
# import wandb
from ot_mmd.utils import get_dist

seed1 = 1102
seed2 = 819
np.random.seed(seed2)
torch.manual_seed(seed2)
torch.cuda.manual_seed(seed2)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--ldas", nargs="+", type=float, default=100)
parser.add_argument("--ohps", nargs="+", type=float, default=-1.0)
parser.add_argument('--method', type=str, default='kluot')

parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n", type=int, default=100, help="number of samples in one set")
parser.add_argument('--start_trial', type=int, default=0)
parser.add_argument('--end_trial', type=int, default=10)
parser.add_argument('--gpu_idx', type=int, default=0)

opt = parser.parse_args()
Tensor = torch.cuda.DoubleTensor if torch.cuda.is_available else torch.DoubleTensor
device = torch.device(f"cuda:{opt.gpu_idx}") if torch.cuda.is_available else torch.device("cpu")

N_per = 100
N1 = opt.n
list_lda = opt.ldas
list_ohp = opt.ohps
method = opt.method

# wandb.login()
# run = wandb.init(
#     project=f"[VAL_P1_{method}]_TST_{ktype}_{case}_{crit}_{max_iter}_{opt.log_msg}")

if isinstance(list_lda, float) or isinstance(list_lda, int):
    list_lda = [list_lda]
else:
    list_lda = list(list_lda)

if isinstance(list_ohp, float) or isinstance(list_ohp, int):
    list_ohp = [list_ohp]
else:
    list_ohp = list(list_ohp)

alpha = 0.05
N = 100
tot_N = 4000


def validate(S_v, list_lda, list_ohp):
    num_correct = {}  # no. of correct by each hp
    nor_margin = {}  # the normalized margin by each hp
    
    best = {"lda": 1, "ohp": 0.1, "nc": None, "nm": None}  # default lambda as 1, sigma for median heuristic
    
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

def run_test(S_v, hp):
    ohp = hp["ohp"]
    lda = hp["lda"]
    
    s1 = S_v[:N1, :]
    s2 = S_v[N1:, :]
    
    s_all = torch.vstack([s1, s2])
    C_all = get_dist(s_all, s_all, p=1)
    
    C = C_all[:s1.shape[0], s1.shape[0]:]
    C = C/C.max()
    
    mu, nu = torch.from_numpy(ot.unif(s1.shape[0])).to(device), torch.from_numpy(ot.unif(s2.shape[0])).to(device)
    
    orig_value = klot(mu, nu, C, ohp, lda, method='sinkhorn_stabilized').item()
    perm_vals = []
    nxy = S_v.shape[0]
    for r in range(N_per):
        ind = np.random.choice(nxy, nxy, replace=False)
        indx, indy = ind[:N1], ind[N1:]
        
        C_per = C_all[np.ix_(indx, indy)]
        C_per = C_per/C_per.max()
        
        perm_vals.append(klot(mu, nu, C_per, ohp, lda, method='sinkhorn_stabilized').item())
        
    perm_vals = np.sort(perm_vals)
    threshold = perm_vals[np.int32(np.ceil(N_per * (1 - alpha)))]
    h = 1 if orig_value > threshold else 0
    return h, threshold, orig_value

def get_hp(kk, method):
    if "kl" in method:
        if kk == 2:
            return {"lda": 10, "ohp": 0.001}
        elif kk == 9:
            return {"lda": 0.1, "ohp": 0.1}
        return {"lda": 1, "ohp": 0.1}

dataloader_FULL = DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=60000,
    shuffle=True,
)

for i, (imgs, Labels) in enumerate(dataloader_FULL):
    data_all = imgs
data_all = data_all.to(device)

Fake_MNIST = pickle.load(open('./Fake_MNIST_data_EP100_N10000.pckl', 'rb'))
ind_all = np.arange(tot_N)
ind_M_all = np.arange(tot_N)

score_trial = []
ntrials = opt.end_trial-opt.start_trial+1

print(f"Method: {method}. n: {opt.n}")

for kk in range(opt.start_trial, opt.end_trial):
    res = 0
    
    torch.manual_seed(kk * 19 + N1)
    torch.cuda.manual_seed(kk * 19 + N1)
    np.random.seed(seed=seed2 * (kk + 9) + N1)

    # 1)--with the seeds for the trial, sample indices train-test

    # load real mnist data
    ind_M_tr = np.random.choice(tot_N, N1, replace=False)
    ind_M_te = np.delete(ind_M_all, ind_M_tr)

    # load fake mnist data

    ind_tr = np.random.choice(tot_N, N1, replace=False)
    ind_te = np.delete(ind_all, ind_tr)

    # 2)--with the sampled indices for train-test, get train & test MNIST data

    Fake_MNIST_tr = torch.from_numpy(Fake_MNIST[0][ind_tr]).to(device)
    Fake_MNIST_te = torch.from_numpy(Fake_MNIST[0][ind_te]).to(device)

    np.random.seed(seed=seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)

    # 3)--The above seeds seem useless

    # Run 2-sample test on training set
    # fetch training data
    s1 = data_all[ind_M_tr]
    s2 = Fake_MNIST_tr.type(Tensor)
    S = torch.cat([s1, s2], dim=0)  # NOTE: removed .cpu()
    S_v = S.view(2*N1, -1)

    best_hp = get_hp(kk, method)  # validate(S_v, list_lda, list_ohp)

    np.random.seed(seed1)
    for k in range(N):  # NOTE: changed their seed
        # 4)--With the seed for trial index, dataset index; sample test indices for both real, fake
        np.random.seed(seed=seed1*(k+1) + 2*kk + N1)
        ind_M = np.random.choice(len(ind_M_te), N1, replace=False)
        s1 = data_all[ind_M_te[ind_M]]

        np.random.seed(seed=seed2*(k+3) + 2*kk + N1)
        ind_F = np.random.choice(len(Fake_MNIST_te), N1, replace=False)
        s2 = Fake_MNIST_te[ind_F].type(Tensor)

        S = torch.cat([s1, s2], dim=0)
        S_v = S.view(2*N1, -1)

        h, thr, val = run_test(S_v, best_hp)
        
        res += h
    score_trial.append(res)
    print(f"--------n {opt.n}, trial {kk}, trial-score {score_trial[-1]}")
print({"n":opt.n, "mean across trials": np.sum(score_trial)/(ntrials*N)})
