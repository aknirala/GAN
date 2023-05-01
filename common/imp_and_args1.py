#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Code adpated from: https://github.com/soumith/dcgan.torch/blob/master/main.lua
import os, sys
import random

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torchvision import transforms
import torchvision.utils as vutils

from tqdm import tqdm
from timeit import default_timer as timer
#import gc

def isIPython():
    try:
        get_ipython().__class__.__name__
        return True
    except:
        return False

ntbk = isIPython()
disp_itrs = [0, 1, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000]

f_name = ""

import ipynbname
log_stuff = False  #Kept as False as while running it is not saved...
if ntbk:    
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))
    f_name = ipynbname.name()
else:  #Not a jupyter notebook... likely running on server
    log_stuff = True
    #Assuming that there is only one dot in the file name.
    f_name = os.path.basename(__file__).split(".")[0]

try:
    ntbk = o_ride_is_ntbk
except:
    pass

try:
    log_stuff = o_ride_do_log_stuff
except:
    pass


#ntbk = False #Overriding it here
#log_stuff = True

#%config Completer.use_jedi = False
#Now some meta code etc [Copy paste always]
cuda = torch.cuda.is_available()
if cuda:
    gpu = 0 #with time would be modifed to use multiple GPUs

from datetime import datetime as dt
run_tStamp = dt.now().strftime("_%m%d_%H%M%S")
#This op_folder will mostly be used for saving images... But 
#  if save_tstamp is set to True then it will be created even for jupyter
op_folder = "OP_"+f_name+run_tStamp


# In[2]:


import argparse
parser = argparse.ArgumentParser()
#dataset argument would be loaded in time
parser.add_argument('--adam', action='store_true',
                    help='Whether to use adam (default is rmsprop)')
parser.add_argument('--batchSize',    type=int, default=64, 
                    help='input batch size')
parser.add_argument('--beta1',        type=float, default=0.5, 
                    help='momentum term of adam')
parser.add_argument('--beta2', type=float, default=0.9, 
                    help='value of beta2.')

parser.add_argument('--clamp_lower',  type=float, default=-0.01)
parser.add_argument('--clamp_upper',  type=float, default=0.01)
parser.add_argument('--critic_iters', type=int,   default=5, 
                    help='# of critic iterations.')

parser.add_argument('--dataroot', default="/home/aknirala/data/lsun", 
                    help='path to dataset')
parser.add_argument('--dataset', default="diverse_clocks", 
                    help='imagenet / lsun / diverse_clock / paired_clocks')
parser.add_argument('--dim',          type=int,   default=64, 
                    help='in gen start with 8*dim channels.')

parser.add_argument('--fineSize',     type=int,   default=64, 
                    help='fineSize???')

parser.add_argument('--hm_gen_size',          type=int,   default=30, 
                    help='Output of noise emitted by preGen\
                    (maps h and m to noise)!!.')

parser.add_argument('--lambda_value', type=float, default=10.0, 
                    help='value of lambda for gradient penality WGAN_GP.')
parser.add_argument('--loadSize',     type=int,   default=96, 
                    help='loadSize???')
parser.add_argument('--lrD',          type=float, default=0.0002, 
                    help='learning rate for Critic')
parser.add_argument('--lrG',          type=float, default=0.0002, 
                    help='learning rate for Generator')


parser.add_argument('--nc',           type=int, default=3, 
                    help='input image channels')
parser.add_argument('--ndf',          type=int, default=64, 
                    help='#  of discrim filters in first conv layer')
parser.add_argument('--nEpochs',      type=int, default=5, 
                    help='# of epochs')

parser.add_argument('--ngf', type=int, default=64, 
                    help='#  of gen filters in first conv layer')
parser.add_argument('--ngpu'  , type=int, default=1, 
                    help='number of GPUs to use')

parser.add_argument('--niter', type=int, default=25, 
                    help='#  of iterations')

parser.add_argument('--noise', default="normal", 
                    help='uniform / normal')
parser.add_argument('--noCuda'  , action='store_true', 
                    help='disables cuda, even if present')
parser.add_argument('--notLog', action='store_true', 
                    help='if passed, no logging will happen.\
                    Logging is off for jupyter anyways.;')

parser.add_argument('--nz', type=int, default=100, 
                    help='size of the latent z vector')

parser.add_argument('--opFolder', default='', 
                    help='Override the logging folder.')

parser.add_argument('--resumeFldr', default='', 
                    help='If passed, from the specified folder\
                    these details will be picked: Latest weight,\
                    iter_count.')

parser.add_argument('--seed', type=int, default=1, 
                    help='Seed for pyTorcj.')
parser.add_argument('--workers', type=int,  default=8, 
                    help='number of data loading workers')


# In[ ]:





# In[ ]:




