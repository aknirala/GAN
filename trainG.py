#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Simply train the G by accounting for the noise type...


"""

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




if ntbk:
    sys.argv = ['main', "--batchSize", "4",
                '--niter', '100', "--noise", "uniform"]#"--mergePreGen"]#,,  "--loadPreGPath", "preG.mdl" ]#, ]#, "--critic_iters", "5"] , '--useLN'
print(sys.argv)
#Now add some parser argument here, commented examples follow
#Boolean, action is what it will be set to when passed
#parser.add_argument('--notLog', action='store_true', help='if passed, no logging will happen. Logging is off for jupyter anyways.;')
#String
#parser.add_argument('--opFolder', default='', help='Override the logging folder.')
#int
#parser.add_argument('--seed', type=int, default=1, help='Seed for pyTorcj.')
#and float
#parser.add_argument('--clamp_lower', type=float, default=-0.01)



# In[ ]:


opt = parser.parse_args()
print(opt) 
if opt.notLog:
    log_stuff = False
    print("No logging is being done!!")

if ntbk and not log_stuff:
    print("OP folder won't be created as we are in jupyter notebook and logging is off")
elif log_stuff:
    if opt.opFolder != '':
        print("Overriding the name of output folder from: ",op_folder,"to: ",opt.opFolder+run_tStamp, " Not recommended!! ")
        op_folder = opt.opFolder+run_tStamp
    if opt.resumeFldr != "":
        if not os.path.exists(opt.resumeFldr):
            print("Given output folder does not exist! So will create new")
        else:
            op_folder = opt.resumeFldr
            print("Things would be resumed from the folder: ", opt.resumeFldr)
    if not os.path.exists(op_folder):
        os.makedirs(op_folder)
    else:
        if opt.resumeFldr == "":
            print("op_folder: ",op_folder," alrady exist.. things may be ovrwritten")

log_file = os.path.join(op_folder, "LOG_"+f_name+run_tStamp+".log")

#Apparently for logging to work I need to restart the code
import logging
if log_stuff:
    print("doing logging: ",log_file)
    r = logging.basicConfig(filename=log_file,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
#logging.basicConfig(filename='example.log',  level=logging.DEBUG)
def plog(*args):
    if log_stuff:
        logging.info(" ".join([str(t) for t in args]))
    else:
        print(*args)

plog("Arguments are: ", opt)
torch.manual_seed(opt.seed)

if opt.noCuda:
    if cuda:
        print("\nCuda won't be used, though a graphics card is present!! (Not advisable)")

#Now some module specific imports:
#sys.path.append("../")
#from freezed.diverseClkFaces import *

def param_count(ntwk):
    return "{:.3f}M".format(np.sum([np.product(p.shape) for p in ntwk.parameters()])/1000000)

torch.set_default_tensor_type(torch.FloatTensor)


# In[ ]:


#import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from freezed.diverseClkFaces import *
#img = None
class clock_datatset(Dataset):
    def __init__(self, root="/home/aknirala/data/clocks/", transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        #global img
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]
        with open(os.path.join(self.root,
                               str(idx//1000),
                               str(idx%1000000) 
                               +".pkl"), "rb") as f:
            #img = pickle.load(f)
            sample = pickle.load(f).transpose([1, 2, 0])#Image.fromarray(np.uint8(img.transpose([1, 2, 0])*255))
        if self.transform:
            sample = self.transform(sample)
        return sample

class diverse_clock(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
    def __len__(self):
        return 1000000 #It is practically infinite; 1B
    def __getitem__(self, idx):
        sample, h_m = getRandomClock() #To avoid error
        sample = sample.copy()
        if self.transform:
            sample = self.transform(sample)
        return sample, h_m

class paired_clocks(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.h_eye = torch.eye(12)
        self.m_eye = torch.eye(60)
    def __len__(self):
        return 1000000 #It is practically infinite
    def __getitem__(self, idx):
        #But many might get the same state!!!
        frm = (idx%5)*100
        till = frm + idx%100
        st0 = np.random.get_state()
        for i in range(frm, till):
            st0[1][i] += 5
        np.random.set_state(st0)
        sample, h_m = getRandomClock(randSeed=-1)
        h2 = np.random.randint(12)
        m2 = np.random.randint(60)
        sample1 = sample.copy()
        #
        np.random.set_state(st0)
        sample, h_m2 = getRandomClock(randSeed=-1, h=h2, m=m2)
        sample2 = sample.copy()
        #
        #Now, let's encode h and m
        H = self.h_eye[[h_m[0], h_m2[0]]]
        M = self.m_eye[[h_m[1], h_m2[1]]]
        #
        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
            sample = torch.stack([sample1, sample2])
            return sample, H, M
        else:
            return [sample1, sample2], H, M

class same_style_clocks(Dataset):
    def __init__(self, randSeed = 2, transform=None):
        self.transform = transform
        self.randSeed = randSeed
        self.h_eye = torch.eye(12)
        self.m_eye = torch.eye(60)
    def __len__(self):
        return 1000000 #It is practically infinite
    def __getitem__(self, idx):
        idx720 = idx%720
        sample, h_m = getRandomClock(randSeed=self.randSeed, h = idx720//60, m = idx720%12)
        sample = sample.copy()
        H = self.h_eye[h_m[0]]
        M = self.m_eye[h_m[1]]
        #
        if self.transform:
            sample = self.transform(sample)
            return sample, H, M
        else:
            return sample, H, M

if opt.dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(opt.fineSize),
                                transforms.CenterCrop(opt.fineSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
elif opt.dataset == 'CelebA': #it is CelebA
    dataset = dset.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(opt.fineSize),
                               transforms.CenterCrop(opt.fineSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'clocks': #it is CelebA
    dataset = clock_datatset(root=opt.dataroot,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize(opt.fineSize),
                                transforms.CenterCrop(opt.fineSize),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
elif opt.dataset == 'diverse_clocks':
    dataset = diverse_clock(transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize(opt.fineSize),
                                transforms.CenterCrop(opt.fineSize),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
elif opt.dataset == 'paired_clocks':
    dataset = paired_clocks(transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
else:
    plog("No dataset for opt.dataset: ",opt.dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))
plog("Len of iterations per epochs: ",len(dataloader))

g_nz =  opt.nz
fixed_noise = torch.randn(opt.batchSize, g_nz)#, 1, 1)
if cuda:
    fixed_noise = fixed_noise.cuda(gpu)

# In[1]:


import torch.nn as nn
import torch


# In[ ]:


all_img = 0
def generate_image(DsAndGs, itr, reals, loss_dict1, 
                   loss_dict2 = None, fakes = None, 
                   oToPickle = None, imgType=None,
                  filePreFix = ""):
    """
    Generates and saves a plot of the true distribution, 
    the generator, and the
    critic.
    """
    global fixed_noise, all_img, ntbk, op_folder
    D, G, optD, optG = DsAndGs
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(2,2)
    if loss_dict2 is None:
        ax1 = fig.add_subplot(gs[0, :])
    else:
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    for keys in loss_dict1:
        ax1.plot(loss_dict1[keys], label = keys)
    ax1.set_title("Status at: "+str(itr if itr >= 0 else all_img)+" iterations.")
    ax1.legend()
    if loss_dict2 is not None:
        for keys in loss_dict2:
            ax2.plot(loss_dict2[keys], label = keys)
        ax2.legend()
    #
    if imgType == 'hist':
        ax3.hist(reals.reshape(-1).cpu().numpy(), bins=100)
    else:
        ax3.imshow(vutils.make_grid(
                    reals[0: (len(reals)) if len(reals) <=36 else 4].cpu(),
                    padding = 2, nrow = 2, normalize = True
                ).numpy().transpose([1, 2, 0]))
    ax3.set_title("Real images")
    #
    if fakes is None:
        fakes = G(fixed_noise)
    if imgType == 'hist':
        ax4.hist(fakes.reshape(-1).cpu().numpy(), bins=100)
    else:
        ax4.imshow(vutils.make_grid(
                    fakes[0: (len(fakes)) if len(fakes) <=36 else 4].detach().cpu(),
                    padding = 2, nrow = 2, normalize = True
                ).numpy().transpose([1, 2, 0]))
    ax4.set_title("Fake images")
    #
    if ntbk:
        plt.show()
    else:
        plt.savefig(op_folder + '/' + filePreFix + 'frame' + str(all_img) + '.jpg')
        plt.close()
        for fname in loss_dict1:
            with open(op_folder+"/" + filePreFix + fname + ".pkl", "wb") as f:
                pickle.dump(loss_dict1[fname], f)
        if loss_dict2 is not None:
            for fname in loss_dict2:
                with open(op_folder+"/" + filePreFix + fname + ".pkl", "wb") as f:
                    pickle.dump(loss_dict2[fname], f)
        #TODO: Also pickle the weights
        if itr > 100:
            D.zero_grad()
            G.zero_grad()
            torch.save({"D":D.state_dict(),
                        "optD": optD.state_dict()}
                        , op_folder+"/D_"+str(itr)+".mdl")
            torch.save({"G":G.state_dict(),
                        "optG": optG.state_dict()}
                       , op_folder+"/G_"+str(itr)+".mdl")
            if oToPickle is not None:
                torch.save(oToPickle
                       , op_folder+"/" + filePreFix + "oToPickle_"+str(itr)+".mdl")
    all_img += 1


# In[2]:


import glob
import os
def loadWeights(resumeFldr = "", loss_dict = {}):
    global G, D, optG, optD, iters, all_img, plog, cuda
    if resumeFldr != "":
        if not os.path.exists(resumeFldr):
            plog("The folder to resume stuff: ",
                  resumeFldr," doesn't exist! No loading!")
            return
        plog("Trying to load everything from the folder: ", resumeFldr)
        weights_files = glob.glob(os.path.join(resumeFldr,"*_*.mdl"))
        weights_files = [os.path.split(w)[-1][:-4] for w in weights_files]
        d_it = -1
        g_it = -1
        plog("Found following cnadidates files: ", ", ".join(weights_files))
        for fNames in weights_files:
            if fNames[:2] == "D_":
                #I am adduming that epoch won't be concatenated to D
                if str.isnumeric(fNames[2:]):
                    iters = int(fNames[2:])
                    if iters > d_it:
                        d_it = iters
            elif fNames[:2] == "G_":
                #I am adduming that epoch won't be concatenated to D
                if str.isnumeric(fNames[2:]):
                    iters = int(fNames[2:])
                    if iters > g_it:
                        g_it = iters
        iters = min(d_it, g_it)
        if iters != -1:
            if d_it != g_it:
                plog("d_wt and g_wt are different, will attempt to load the min: ", iters)
                plog("If files for both does not exist, code will fail")
                #d_it = iters
                #g_it = iters
            plog("Will load files from iters: ", iters)
            d_chkpt = torch.load(os.path.join(resumeFldr,
                                               "D_" + str(iters) + ".mdl"))
            g_chkpt = torch.load(os.path.join(resumeFldr,
                                               "G_" + str(iters) + ".mdl"))
            if type(d_chkpt) is dict:
                plog("Will load both opt (if not None) and model for D")
                if cuda: D.cuda()
                D.load_state_dict(d_chkpt["D"])
                if optD is not None:
                    optD.load_state_dict(d_chkpt["optD"])
            else:
                plog("Will only load model for D")
                D.load_state_dict(d_chkpt)
            if type(g_chkpt) is dict:
                plog("Will load both opt (if not None) and model for G")
                if cuda: G.cuda()
                G.load_state_dict(g_chkpt["G"])
                if optG is not None:
                    optG.load_state_dict(g_chkpt["optG"])
            else:
                print("Will only load model for G")
                G.load_state_dict(g_chkpt)
            D.train()
            G.train()
            #Since it is possible that losses have different 
            # # of elements depending on what kind of arch it is
            # like for WGAN critic_iters could be more...
            # So, no limiting to iters.
            for fName in loss_dict:
                if not os.path.exists(os.path.join(resumeFldr, fName)):
                    plog("Can't load loss log: ",fName," file doesn't exist")
                else:
                    loss_dict[fName].clear()
                    with open(os.path.join(resumeFldr, fName), "rb") as f:
                        loss_dict[fName] += pickle.load(f)
            frames = glob.glob(os.path.join(resumeFldr,"frame*.jpg"))
            frames = [os.path.split(f)[-1][5:-4] for f in frames]
            all_img = 0
            for frame in frames:
                if str.isnumeric(frame):
                    if int(frame) > all_img:
                        all_img = int(frame)
            all_img += 1
            plog(all_img)
        else:
            iters = -1
    else:
        plog("New folder would be created nothing to resume.")


# In[ ]:






# In[2]:



# In[2]:


import torch.nn as nn


# In[3]:


class MeanPoolConv(nn.Module):
    """
    For conv: padding is same, stride is by default 1
    """
    def __init__(self, name, in_ch, out_ch,
                 filter_size, he_init=True, bias=True):
        super(MeanPoolConv, self).__init__()
        self.main = nn.Sequential()
        avg_pool = nn.AvgPool2d(2)
        conv = nn.Conv2d(in_ch, out_ch, filter_size, 
                         padding = (filter_size - 1)//2, bias=bias)
        if he_init:
            nn.init.kaiming_uniform_(conv.weight)
        self.main.add_module(name + "_avg_pool", avg_pool)
        self.main.add_module(name + "_conv", conv)
    def forward(self, inputs):
        return self.main(inputs)

class ConvMeanPool(nn.Module):
    """
    For conv: padding is same, stride is by default 1
    """
    def __init__(self, name, in_ch, out_ch,
                 filter_size, he_init=True, bias=True):
        super(ConvMeanPool, self).__init__()
        self.main = nn.Sequential()
        conv = nn.Conv2d(in_ch, out_ch, filter_size, 
                         padding = (filter_size - 1)//2, bias=bias)
        if he_init:
            nn.init.kaiming_uniform_(conv.weight)
        avg_pool = nn.AvgPool2d(2)
        self.main.add_module(name + "_conv", conv)
        self.main.add_module(name + "_avg_pool", avg_pool)
    def forward(self, inputs):
        return self.main(inputs)

class UpsampleConv(nn.Module):
    """
    For conv: There is no padding here, stride is by default 1
    """
    def __init__(self, name, in_ch, out_ch,
                 filter_size, he_init=True, bias=True):
        super(UpsampleConv, self).__init__()
        self.main = nn.Sequential()
        up_sample = nn.UpsamplingNearest2d(scale_factor = 2)
        conv = nn.Conv2d(in_ch, out_ch, filter_size, 
                         padding = (filter_size - 1)//2, bias=bias)
        if he_init:
            nn.init.kaiming_uniform_(conv.weight)
        self.main.add_module(name + "_up_sample", up_sample)
        self.main.add_module(name + "_conv", conv)
    def forward(self, inputs):
        return self.main(inputs)


# In[4]:


# Residual block Down
class ResidualBlockDown(nn.Module):
    def __init__(self, name, in_ch, out_ch, size, k_size = 3, stride=1, 
                 l_norm = True):
        """
        resample = down means ResidualBlock is being used in discriminator
        So we will use LayerNorm instead of batch_norm in discriminator.
        
        Here is what is happening (for down):
        i/p --> || Normalize, ReLU, Conv1 (no channel change) || 
         |    --> Normalize, ReLU, ConvMeanPool (Sz half & channel change)
         |_=> MeanPoolConv (Channel chane & sz half)
        
        """
        super(ResidualBlockDown, self).__init__()
        self.s_cut = MeanPoolConv(name + "_down_mpc", in_ch, out_ch, k_size)
        if l_norm:
            self.nmlz1 = nn.LayerNorm([in_ch, size, size])
            self.nmlz2 = nn.LayerNorm([in_ch, size, size])
        else:
            #We would use batch norm
            self.nmlz1 = nn.BatchNorm2d(in_ch)
            self.nmlz2 = nn.BatchNorm2d(in_ch)
        #
        #Now we would reach the same dimension as reached by 
        #      self.shortcut via two convolutinal loops.
        #  self.shortcut would reach: [B, O_Ch, W//2, W//2]
        #No channel change
        self.R1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, in_ch, k_size,
                              padding = (k_size - 1)//2, bias=False)
        self.R2 = nn.ReLU()
        self.conv2 = ConvMeanPool(name + "_down_cmp", in_ch, out_ch, k_size)
        
    def forward(self, x):
        s_cut = self.s_cut(x)
        out = self.R1(self.nmlz1(x))
        out = self.conv1(out)
        #out = self.R2(out)
        out = self.R2(self.nmlz2(out))
        out = self.conv2(out)
        return out + s_cut

# Residual block Up
class ResidualBlockUp(nn.Module):
    def __init__(self, name, in_ch, out_ch, k_size = 3, stride=1):
        super(ResidualBlockUp, self).__init__()
        self.s_cut = UpsampleConv(name + "_up_sample_s_cut", 
                                  in_ch, out_ch, k_size)
        self.nmlz1 = nn.BatchNorm2d(in_ch)
        self.nmlz2 = nn.BatchNorm2d(out_ch)
        #
        #Now we would reach the same dimension as reached by 
        #      self.shortcut via two convolutinal loops.
        #  self.shortcut would reach: [B, O_Ch, W//2, W//2]
        #No channel change
        self.R1 = nn.ReLU()
        self.conv1 = UpsampleConv(name + "_up_sample_conv1_", 
                                  in_ch, out_ch, k_size, bias=False)
        self.R2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, k_size,
                              padding = (k_size - 1)//2)
        
    def forward(self, x):
        s_cut = self.s_cut(x)
        out = self.R1(self.nmlz1(x))
        out = self.conv1(out)
        out = self.R2(self.nmlz2(out))
        out = self.conv2(out)
        return out + s_cut


# In[5]:


class GoodGenerator(nn.Module):
    def __init__(self, nz = None):
        super(GoodGenerator, self).__init__()
        global opt
        #.view would be used in forward
        
        self.L = nn.Linear(opt.nz if nz is None else nz, 4*4*8*opt.dim)
        self.R1 = ResidualBlockUp('G_R1', 8*opt.dim, 8*opt.dim)
        self.R2 = ResidualBlockUp('G_R2', 8*opt.dim, 4*opt.dim)
        self.R3 = ResidualBlockUp('G_R3', 4*opt.dim, 2*opt.dim)
        self.R4 = ResidualBlockUp('G_R4', 2*opt.dim, opt.dim)
        self.norm = nn.BatchNorm2d(opt.dim)
        self.relu = nn.ReLU()
        #Below, we don't need to use formula for padding as kernel size 
        #    is being decided over here.
        self.conv2rgb = nn.Conv2d(opt.dim, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        global opt
        out = self.L(x).view(-1, 8*opt.dim, 4, 4)
        out = self.R1(out)
        out = self.R2(out)
        out = self.R3(out)
        out = self.R4(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv2rgb(out)
        return self.tanh(out)


# In[6]:


class GoodDiscriminator(nn.Module):
    def __init__(self):
        super(GoodDiscriminator, self).__init__()
        global opt
        self.rgb2conv = nn.Conv2d(3, opt.dim, kernel_size=3, padding=1)
        self.R1 = ResidualBlockDown('D_R1', opt.dim, 2*opt.dim, 64)
        self.R2 = ResidualBlockDown('D_R2', 2*opt.dim, 4*opt.dim, 32)
        self.R3 = ResidualBlockDown('D_R3', 4*opt.dim, 8*opt.dim, 16)
        self.R4 = ResidualBlockDown('D_R4', 8*opt.dim, 8*opt.dim, 8)
        self.L = nn.Linear(4*4*8*opt.dim, 1)
    def forward(self, x):
        global opt
        out = self.rgb2conv(x)
        out = self.R1(out)
        out = self.R2(out)
        out = self.R3(out)
        out = self.R4(out)
        #print(out.shape, opt.dim)
        return self.L(out.view(-1, 4*4*8*opt.dim))


# In[7]:


def calc_gradient_penalty(netD, real_data, fake_data):
    """
    This is voodoo of gradient penalty which makes everything glitter.
    """
    global opt
    alpha = torch.rand(real_data.shape[0], 1) #This is b/w 0 and 1
    #alpha = alpha.expand(real_data.size())
    alpha = alpha.expand(real_data.shape[0], 
                         real_data.nelement()//real_data.shape[0]).contiguous().view(
                        *real_data.shape)
    alpha = alpha.cuda(gpu) if cuda else alpha
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    if cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs      = disc_interpolates, 
                              inputs       = interpolates,
                              grad_outputs = torch.ones(
                                                  disc_interpolates.size()
                                              ).cuda() if cuda else torch.ones(
                                                  disc_interpolates.size()),
                              create_graph = True,
                              retain_graph = True, 
                              #only_inputs  = True #This is True by default.
                             )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda_value
    #I believe .mean() is not needed
    #gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2) * LAMBDA
    return gradient_penalty


# In[ ]:


def do_WPG_iters(reals, noises, dgs, losses):
    """
    This will perform the criitc iterations while doing WGAN_PG training. and then G iter
    reals: an array of real data
    noise: an array of noise. It is 1 more than reals in size (for G iteration)
    dgs: an array containing [G, D, optG, optD], in order
    losses: an array fo rall the losses expected while doing WGAN_PG training
            so, the array would be in order: [R_COST, F_COST, GP, W_DIST, D_COST, G_COST]
            
    """
    #Static part are made global
    global one, mone
    G, D, optG, optD = dgs
    R_COST, F_COST, GP, W_DIST, D_COST, G_COST = losses
    for p in D.parameters():
        p.requires_grad = True
    for c_iters in range(len(reals)):
        D.zero_grad()
        #
        #Train with reals
        data = reals[c_iters].cuda(gpu)
        D_real = D(data.float())
        D_real = D_real.mean().reshape(1)
        D_real.backward(one)
        #
        #Train with fakes
        noise = noises[c_iters].cuda(gpu)
        with torch.no_grad():
            fakes = G(noise).data
        D_fake = D(fakes)
        D_fake = D_fake.mean().reshape(1)
        D_fake.backward(mone)
        #
        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(D, data.data, fakes.data)
        gradient_penalty.backward()
        optD.step()
        #--------------Now append losses------------
        
        R_COST.append(float(D_real))
        F_COST.append(float(D_fake))
        GP.append(float(gradient_penalty))
        Wasserstein_D = D_real - D_fake
        W_DIST.append(float(Wasserstein_D))
        D_cost = D_fake - D_real + gradient_penalty
        D_COST.append(float(D_cost))
    #
    ############################
    # (2) Update G network
    ###########################
    for p in D.parameters():
        p.requires_grad = False  # to avoid computation
    G.zero_grad()
    #
    noise = noises[-1].cuda(gpu)
    fakes = G(noise)
    G_cost = D(fakes)
    G_cost = G_cost.mean().reshape(1)
    G_cost.backward(one)
    optG.step()
    G_COST.append(float(G_cost))



# In[3]:


G = GoodGenerator(opt.nz)
plog(G)
D = GoodDiscriminator()
plog(D)


# In[4]:


#Now declare the variables!!
optD = optim.Adam(D.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
optG = optim.Adam(G.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
optGDec = optim.Adam(G.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
R_COST = []
F_COST = []
W_DIST = []
D_COST = []
G_COST = []
GP = []

iters = -1
try:
    loadWeights(opt.resumeFldr, loss_dict={
                        "R_COST.pkl": R_COST,
                        "F_COST.pkl": F_COST,
                        "W_DIST.pkl": W_DIST,
                        "D_COST.pkl": D_COST,
                        "G_COST.pkl": G_COST,
                    })
except:
    plog("Encountered error while loading weights!!")

one = torch.FloatTensor([1])
mone = one * -1
if cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)
    D = D.cuda(gpu)
    G = G.cuda(gpu)


# In[5]:


data_len = len(dataloader)
D_real = 1
D_fake = -1
gradient_penalty = opt.lambda_value
G_cost = 1
loss = 1
for epoch in range(opt.nEpochs):
    plog("Epoch: ", epoch, " of ", opt.nEpochs, " iters: ", iters)
    data_iter = iter(dataloader)
    
    i = 0
    pbar = tqdm(total=len(data_iter)//opt.critic_iters)
    #pbar = tqdm(range(len(data_iter)//opt.critic_iters))
    while i < data_len - opt.critic_iters:
        pbar.set_description( (" D_real {:.3f}, D_fake {:.3f}, "        +  "gradient_penalty {:.3f}, G_cost {:.3f}, {}").format(
            float(D_real), float(D_fake), 
            float(gradient_penalty), float(G_cost), iters))
        pbar.update(1)
        i += 5
        iters += 1
        reals = []
        noises = []
        for iter_d in range(opt.critic_iters):
            reals.append(next(data_iter)[0])
            if opt.noise == "uniform":
                noises.append(torch.FloatTensor(opt.batchSize, g_nz).uniform_(-1, 1))
            else:
                noises.append(torch.randn(opt.batchSize, g_nz))
        noises.append(torch.randn(opt.batchSize, g_nz))
        do_WPG_iters(reals, noises, 
                     [G, D, optG, optD], 
                     [R_COST, F_COST, GP, W_DIST, D_COST, G_COST])
        
        if iters%5000 == 0 or iters in disp_itrs:
            generate_image([D, G, optD, optG],
                           iters, reals[-1], 
                           loss_dict1 = {"R_COST":R_COST,
                                        "F_COST":F_COST,
                                        "W_DIST":W_DIST,
                                        #"D_COST":D_COST,
                                        "GP":GP}, 
                           loss_dict2 = {"G_COST":G_COST
                                        })#,
                           #oToPickle = oToPickle)#, fakes.data)
        #if ntbk and iters > 500:
        #    break
    pbar.close()
#with batch


# In[ ]:




