import argparse
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import ot
from ot_mmd.utils import get_dist

seed1 = 1102
seed2 = 819
np.random.seed(seed2)
torch.manual_seed(seed2)
torch.cuda.manual_seed(seed2)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--ohps", nargs="+", type=float, default=-1.0)
parser.add_argument('--ktype', type=str, default='rbf')
parser.add_argument('--method', type=str, default='MMD')

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
list_ohp = opt.ohps
ktype = opt.ktype
method = opt.method

if isinstance(list_ohp, float) or isinstance(list_ohp, int):
    list_ohp = [list_ohp]
else:
    list_ohp = list(list_ohp)

list_ohp = list_ohp + [None]  # for median heuristic

alpha = 0.05
N = 100
tot_N = 4000

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

def get_val(s1_t, s2_t, hp, G=None, indx=None, indy=None, call1=0):
    # returns obj value for a given hp
    ohp = hp["ohp"]
    
    if G is None:
        data_cat = torch.vstack([s1_t, s2_t])
        G = get_rbf_G(khp=ohp, x=data_cat, y=data_cat)
        indx = np.arange(s1_t.shape[0])
        indy = np.arange(s1_t.shape[0], data_cat.shape[0])
    
    G1 = G[np.ix_(indx, indx)]
    G2 = G[np.ix_(indy, indy)]
    G12 = G[np.ix_(indx, indy)]
    
    a = torch.from_numpy(ot.unif(s1_t.shape[0])).to(device)
    b = torch.from_numpy(ot.unif(s2_t.shape[0])).to(device)

    val = np.sqrt((a.dot(torch.mv(G1, a)) + b.dot(torch.mv(G2, b)) - 2*a.dot(torch.mv(G12, b))).item())
    if call1:
        return val, G
    return val

def run_test(S_v, hp):
    s1 = S_v[:N1, :]
    s2 = S_v[N1:, :]
    
    orig_value, G = get_val(s1, s2, hp, call1=1)
    perm_vals = []
    nxy = S_v.shape[0]
    for r in range(N_per):
        ind = np.random.choice(nxy, nxy, replace=False)
        indx, indy = ind[:N1], ind[N1:]
        
        perm_vals.append(get_val(S_v[indx], S_v[indy], hp, G, indx, indy))
        
    perm_vals = np.sort(perm_vals)
    threshold = perm_vals[np.int32(np.ceil(N_per * (1 - alpha)))]
    h = 1 if orig_value > threshold else 0
    return h, threshold, orig_value

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

def get_hp(kk, method):
    if method == "MMD":
        return {"lda": -1, "ohp": -1}

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

    # best_hp = get_hp(kk, method)  # uncomment this to directly get the validated hp
    best_hp = validate(S_v, [-1], list_ohp)  # TODO: comment this.
    
    # print(best_hp)

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
