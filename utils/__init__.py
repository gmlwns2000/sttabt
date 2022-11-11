from utils.threadbuffer import ThreadBuffer
from utils.query_available_devices import query_available_devices

import torch

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x

def mkdir(path, create_message=None):
    import os
    
    if os.path.exists(path):
        return
    
    import pathlib
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    if create_message is not None:
        print(create_message)

def initialize_saves_dirs():
    mkdir('./saves')
    mkdir('./saves/vit_concrete_end2end_logs')
    mkdir('./saves/lvvit', create_message=
        '[SETUP] LVViT weight must be downloaded seperately! into ./saves/lvvit/*\n'+\
        'wget https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar\n'+\
        'from https://github.com/zihangJiang/TokenLabeling'
    )
    mkdir('./saves_plot')
    mkdir('./saves_hparam')
    mkdir('./logs')