from ot_mmd.barycenter import solve_apgd
from ot_mmd.utils import createLogHandler, get_t, get_dist, get_G
import os
import argparse
import joblib
import torch

parser = argparse.ArgumentParser(description="_")
parser.add_argument("--t_pred", required=True, type=int)
parser.add_argument("--best_lda", type=float, default=None)
parser.add_argument("--best_hp", type=float, default=None)
parser.add_argument("--save_as", default="")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
dtype = torch.float64
max_itr = 1000
ktype = "imq_v2"
t_predict = args.t_pred

logger = createLogHandler(f"{args.save_as}.csv", str(os.getpid()))

if args.best_lda is None:
    valt_predict = list(set([1, 2, 3]).symmetric_difference(set([t_predict])))
    best_score = torch.inf
    val = {}
    for lda in [10, 1e-1, 1]:
        val[lda] = {}
        for khp in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, None]:
            val[lda][khp] = []
            for t in valt_predict:
                init_tstep = t-1
                final_tstep = t+1

                data_tpredict = get_t(joblib.load(f"data/EB_t{t}.pickle"), device=device)
                
                data_init = get_t(joblib.load(f"data/EB_t{init_tstep}.pickle"), device=device)
                data_final = get_t(joblib.load(f"data/EB_t{final_tstep}.pickle"), device=device)

                data_all = torch.vstack([data_init, data_final])
                C = {1: get_dist(x=data_init, y=data_all, p=1),
                    2: get_dist(x=data_final, y=data_all, p=1)}
                
                G_all = get_G(ktype=ktype, khp=khp, x=data_all, y=data_all)
                m1 = data_init.shape[0]
                G = {1: G_all[:m1, :m1], 2: G_all[m1:, m1:], 'all': G_all}
                
                a = (torch.ones(data_init.shape[0])/data_init.shape[0]).to(dtype).to(device)
                b = (torch.ones(data_final.shape[0])/data_final.shape[0]).to(dtype).to(device)
                
                bary, _ = solve_apgd(C, G, {1: a, 2: b}, max_itr, {1: lda, 2: lda}, case="bal")
                
                gt = (torch.ones(data_tpredict.shape[0])/data_tpredict.shape[0]).to(dtype).to(device)
                data_cat = torch.vstack([data_tpredict, data_all])
                G = get_G(ktype="rbf", x=data_cat, y=data_cat)
                vec = torch.cat([gt, -bary])
                val[lda][khp].append(torch.mv(G, vec).dot(vec).item())
                
            logger.info(f", {lda}, {khp}, {sum(val[lda][khp])}")
            if sum(val[lda][khp]) < best_score:
                best_score = sum(val[lda][khp])
                best_config = {"lda": lda, "khp": khp}
        
    lda = best_config["lda"]
    khp = best_config["khp"]
else:
    lda = args.best_lda
    khp = args.best_hp

t = t_predict

init_tstep = t-1
final_tstep = t+1

data_tpredict = get_t(joblib.load(f"data/EB_t{t}.pickle"), device=device)

data_init = get_t(joblib.load(f"data/EB_t{init_tstep}.pickle"), device=device)
data_final = get_t(joblib.load(f"data/EB_t{final_tstep}.pickle"), device=device)

data_all = torch.vstack([data_init, data_final])
C = {1: get_dist(x=data_init, y=data_all, p=1),
     2: get_dist(x=data_final, y=data_all, p=1)}

G_all = get_G(ktype=ktype, khp=khp, x=data_all, y=data_all)
m1 = data_init.shape[0]
G = {1: G_all[:m1, :m1], 2: G_all[m1:, m1:], 'all': G_all}

a = (torch.ones(data_init.shape[0])/data_init.shape[0]).to(dtype).to(device)
b = (torch.ones(data_final.shape[0])/data_final.shape[0]).to(dtype).to(device)

bary, _ = solve_apgd(C, G, {1: a, 2: b}, max_itr, {1: lda, 2: lda}, case="bal")

gt = (torch.ones(data_tpredict.shape[0])/data_tpredict.shape[0]).to(dtype).to(device)
data_cat = torch.vstack([data_tpredict, data_all])
G = get_G(ktype="rbf", x=data_cat, y=data_cat)
vec = torch.cat([gt, -bary])
val_chosen = torch.sqrt(torch.mv(G, vec).dot(vec)).item()
logger.info(f"UOT-MMD, {t}, {val_chosen}")
