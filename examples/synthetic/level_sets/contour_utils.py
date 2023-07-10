import numpy as np
import ot
import matplotlib.pyplot as plt


def get_square_indices(dat):
    ix = []
    ix.append(dat)
    ix.append(tuple(-1*np.array(dat)))
    ix.append(tuple(np.array([-1, 1])*np.array(dat)))
    ix.append(tuple(np.array([1, -1])*np.array(dat)))
    return ix

def mirrored(maxval, inc=1):
    x = np.arange(inc, maxval, inc)
    if x[-1] != maxval:
        x = np.r_[x, maxval]
    return np.r_[-x[::-1], 0, x]

def get_data(d, intv):
    x = mirrored(d, intv)
    y = mirrored(d, intv)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')    
    nx = x.shape[0]
    ny = y.shape[0]
    data = []
    for i in range(nx):
        for j in range(ny):
            data.append([xv[i, j], yv[i, j]])
    data = np.array(data)
    return data

def get_distr_Q(dq, data_Q):
    distr_Q = np.zeros(data_Q.shape[0])
    ix = [np.where((data_Q == (-dq, -dq)).all(axis = 1))[0][0], np.where((data_Q == (dq, dq)).all(axis = 1))[0][0]]
    distr_Q[ix] = 0.5
    return distr_Q 

def get_distr_P(data_P, dat):
    distr_P = np.zeros(data_P.shape[0])
    ix = [np.where((data_P == dat).all(axis = 1))[0][0], np.where((data_P == tuple(-1*np.array(dat))).all(axis = 1))[0][0]]
    distr_P[ix] = 0.5
    if dat == (0, 0):
        ix = np.where(distr_P>0)
        distr_P[ix] = 1
    return distr_P

def get_emd(M, wa, wb):
    G = ot.emd(wa, wb, M)
    dist = np.sum(G * M)
    return dist, G

def plot_fn(xv, yv, Z, intv, save_as, tot_l = 10):
    fig, ax = plt.subplots(1, 1)
    cpf = ax.contourf(xv, yv, Z, tot_l, cmap="hot")
    colours = ['w' if level<0 else 'k' for level in cpf.levels]
    ax.contour(xv, yv, Z, tot_l, colors = colours, linewidths = 0.5)
    plt.colorbar(cpf)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end+intv, intv))
    ax.yaxis.set_ticks(np.arange(start, end+intv, intv))
    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1)
    plt.show()
