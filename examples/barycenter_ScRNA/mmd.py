from ot_mmd.utils import createLogHandler, get_t, get_G
import os
import argparse
import joblib
import torch

parser = argparse.ArgumentParser(description="_")
parser.add_argument("--t_pred", required=True, type=int)
parser.add_argument("--save_as", default="")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
dtype = torch.float64
t_predict = args.t_pred

logger = createLogHandler(f"{args.save_as}.csv", str(os.getpid()))

t = t_predict

init_tstep = t-1
final_tstep = t+1

data_tpredict = get_t(joblib.load(f"data/EB_t{t}.pickle"), device=device)

data_init = get_t(joblib.load(f"data/EB_t{init_tstep}.pickle"), device=device)
data_final = get_t(joblib.load(f"data/EB_t{final_tstep}.pickle"), device=device)

data_all = torch.vstack([data_init, data_final])

a = (torch.ones(data_init.shape[0])/data_init.shape[0]).to(dtype).to(device)
b = (torch.ones(data_final.shape[0])/data_final.shape[0]).to(dtype).to(device)

bary = torch.cat([a, b])/2

gt = (torch.ones(data_tpredict.shape[0])/data_tpredict.shape[0]).to(dtype).to(device)
data_cat = torch.vstack([data_tpredict, data_all])
G = get_G(ktype="rbf", x=data_cat, y=data_cat)
vec = torch.cat([gt, -bary])
val_chosen = torch.sqrt(torch.mv(G, vec).dot(vec)).item()
logger.info(f"Method, tstep, MMD (lower is better)")
logger.info(f"MMD, {t}, {val_chosen}")
