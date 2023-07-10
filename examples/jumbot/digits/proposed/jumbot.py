import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import itertools
import torch.nn.functional as F
import pdb
from ot_mmd.mmdot import solve_apgd
from ot_mmd.utils import get_G, get_t
import ot
import os
# import wandb
from jumbot_utils import model_eval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    import os
    os.environ['main_phd'] = str(seed)

set_seed(1980)

class Jumbot(object):
    """Jumbot class"""
    def __init__(self, model_g, model_f, n_class, reg_type, lda, max_itr, khp, verbose, ktype, eta1, eta2, case, ridge, wd, crit, save_as=""):
        """
        Initialize jumbot method.
        args :
        - model_g : feature exctrator (torch.nn)
        - model_f : classification layer (torch.nn)
        - n_class : number of classes (int)
        - eta_1 : feature comparison coefficient (float)
        - eta_2 : label comparison coefficient (float)
        - lda : marginal coeffidient (float)
        - epsilon : entropic regularization (float)
        """
        self.save_as = save_as
        self.model_g = model_g   # target model
        self.model_f = model_f
        self.n_class = n_class
        self.eta1 = eta1  # weight for the alpha term
        self.eta2 = eta2 # weight for target classification
        self.lda = lda
        self.khp = khp
        self.ktype = ktype 
        self.reg_type = reg_type
        self.verbose = verbose
        self.max_itr = max_itr
        self.case = case
        self.ridge = ridge
        self.crit = crit
        self.wd = wd
    
    def fit(self, source_loader, target_loader, test_loader, n_epochs, criterion=nn.CrossEntropyLoss()):
        """
        Run jumbot method.
        args :
        - source_loader : source dataset 
        - target_loader : target dataset
        - test_loader : test dataset
        - n_epochs : number of epochs (int)
        - criterion : source loss (nn)
        
        return:
        - trained model
        """
        target_loader_cycle = itertools.cycle(target_loader)
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=2e-4)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=2e-4)
        
        for id_epoch in range(n_epochs):
            
            self.model_g.train()
            self.model_f.train()
            
            for i, data in enumerate(source_loader):
                # print('___batchid_{}'.format(i))
                ### Load data
                
                xs_mb, ys = data
                xs_mb, ys = xs_mb.to(device), ys.to(device)
                xt_mb, _ = next(target_loader_cycle)
                xt_mb = xt_mb.to(device)
                
                g_xs_mb = self.model_g(xs_mb)
                f_g_xs_mb = self.model_f(g_xs_mb)
                g_xt_mb = self.model_g(xt_mb)
                f_g_xt_mb = self.model_f(g_xt_mb)
                pred_xt = F.softmax(f_g_xt_mb, 1)
                
                ### loss
                s_loss = criterion(f_g_xs_mb, ys)

                ###  Ground cost
                embed_cost = torch.cdist(g_xs_mb, g_xt_mb)**2
                
                ys = F.one_hot(ys, num_classes=self.n_class).float()
                t_cost = - torch.mm(ys, torch.transpose(torch.log(pred_xt), 0, 1))
                
                total_cost = self.eta1 * embed_cost + self.eta2 * t_cost


                detached_gxs = g_xs_mb.detach().to(total_cost.dtype)
                detached_gxt = g_xt_mb.detach().to(total_cost.dtype)
                cost = total_cost.detach()

                G1 = get_G(x=detached_gxs, y=detached_gxs, ktype=self.ktype, khp=self.khp, ridge=self.ridge)
                G2 = get_G(x=detached_gxt, y=detached_gxt, ktype=self.ktype, khp=self.khp, ridge=self.ridge)

                #OT computation
                a, b = get_t(ot.unif(g_xs_mb.size()[0]), device=device, dtype=total_cost.dtype), get_t(ot.unif(g_xt_mb.size()[0]), device=device, dtype=total_cost.dtype)
                
                pi, _ = solve_apgd(cost, {1: G1, 2: G2}, {1: a, 2: b}, self.max_itr, self.lda, case=self.case, crit=self.crit)
                
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                
                da_loss = torch.tensordot(pi, total_cost)
                tot_loss = s_loss + da_loss
                tot_loss.backward()
                
                optimizer_g.step()                
                optimizer_f.step()

            # print('epoch, loss : ', id_epoch, s_loss.item(), da_loss.item())
            # if id_epoch%10 == 0:
            #     source_acc = self.evaluate(source_loader)
            #     target_acc = self.evaluate(test_loader)
                # wandb.Table(columns=["epoch", "tgt_acc", "source_acc", "lambda", "max_itr", "khp", "ktype", "case", "crit", "reg_type"], 
                            # data=[[id_epoch, target_acc, source_acc, self.lda, self.max_itr, self.khp, self.ktype, self.case, self.crit, self.reg_type]])
                # wandb.log({"epoch": id_epoch, "tgt_acc": target_acc, "src_acc": source_acc})
        torch.save(self.model_g, os.path.join(self.save_as, "model_g.pt"))
        torch.save(self.model_f, os.path.join(self.save_as, "model_f.pt"))
        return tot_loss

    def source_only(self, source_loader, criterion=nn.CrossEntropyLoss()):
        """
        Run source only.
        args :
        - source_loader : source dataset 
        - criterion : source loss (nn)
        
        return:
        - trained model
        """
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=2e-4)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=2e-4)

        for id_epoch in range(10):
            self.model_g.train()
            self.model_f.train()
            for i, data in enumerate(source_loader):
                ### Load data
                xs_mb, ys = data
                xs_mb, ys = xs_mb.to(device), ys.to(device)
                
                g_xs_mb = self.model_g(xs_mb.to(device))
                f_g_xs_mb = self.model_f(g_xs_mb)

                ### loss
                s_loss = criterion(f_g_xs_mb, ys.to(device))

                # train the model 
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                tot_loss = s_loss
                tot_loss.backward()

                optimizer_g.step()
                optimizer_f.step()
        
        return tot_loss
    

    def evaluate(self, data_loader):
        score = model_eval(data_loader, self.model_g, self.model_f)
        return score
