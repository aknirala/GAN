#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch.nn as nn


# In[3]:


import inspect
def weights_init(m):
    """
    It is only doing it for Conv and BatchNorm 
    Noen of that is used in toy example.
    """
    if not str(inspect.getmro(type(m))[0]).startswith("<class 'torch.nn.modules"):
    #if type(m) is DCGANConvLayer:
        #print("Going recursive for: ", str(m)[:100],"==========")
        for L in m.children():
            weights_init(L)
        return
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        try:
            if m.bias is not None:
                print("For DCGAN for Conv(Trans) bias should not be there. But got it for: ",
                      m," classname: ",classname)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class DCGANGenLayer(nn.Module):
    """
    Directly applying batchnorm to all layers however, resulted 
    in sample oscillation and model instability. This was avoided
    by not applying batchnorm to the generator output layer and 
    the discriminator input layer.
    """
    def __init__(self, name, n_in, n_out, k_size = 4, 
                 stride = 1, padding = 0, batch_norm = True
                ):
        super(DCGANGenLayer, self).__init__()
        self.main = nn.Sequential()
        #ConvTranspose2d from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        self.main.add_module(name + "_ConvTrans2d", nn.ConvTranspose2d(n_in, n_out, k_size, stride, padding, bias=False))
        if batch_norm:
            self.main.add_module(name + "_BN", nn.BatchNorm2d(n_out))
        self.main.add_module(name + "_ReLU", nn.ReLU())
        
    def forward(self, inputs):
        return self.main(inputs)


# In[4]:


class DCGAN_Generator(nn.Module):
    def __init__(self, opt):
        super(DCGAN_Generator, self).__init__()
        self.main = nn.Sequential()
        #b_size x 100 x 1 x 1     --->   b_size x 512 x 4 x 4
        self.main.add_module("L1", 
            DCGANGenLayer("Generator_1", opt.nz, opt.ngf*8, 4, 1, 0))
        
        # ---> 512//2 x 8 x 8
        self.main.add_module("L2", 
            DCGANGenLayer("Generator_2", opt.ngf*8, opt.ngf*4, 4, 2, 1))
        
        # ---> 512//4 x 16 x 16
        self.main.add_module("L3", 
            DCGANGenLayer("Generator_3", opt.ngf*4, opt.ngf*2, 4, 2, 1))
        
        # ---> 512//8 = 64 x 32 x 32
        self.main.add_module("L4", 
            DCGANGenLayer("Generator_3", opt.ngf*2, opt.ngf, 4, 2, 1))
        
        #Now 64 channels to opt.nc x 64 x 64
        self.main.add_module("L5", 
            nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False))
        
        self.main.add_module("L6", nn.Tanh())
    def forward(self, noise):
        return self.main(noise)


# In[5]:


class DCGANDiscLayer(nn.Module):
    def __init__(self, name, n_in, n_out, k_size = 4, 
                 stride = 1, padding = 0, batch_norm = True
                ):
        super(DCGANDiscLayer, self).__init__()
        self.main = nn.Sequential()
        #ConvTranspose2d from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        self.main.add_module(name + "_Conv2d", nn.Conv2d(n_in, n_out, k_size, stride, padding, bias = False))
        if batch_norm:
            self.main.add_module(name + "_BN", nn.BatchNorm2d(n_out))
        self.main.add_module(name + "_LeakyReLU", nn.LeakyReLU(0.2))
        
    def forward(self, inputs):
        return self.main(inputs)


# In[6]:


class DCGAN_Discriminator(nn.Module):
    def __init__(self, opt):
        super(DCGAN_Discriminator, self).__init__()
        self.main = nn.Sequential()
        #b_size x 100 x 1 x 1     --->   b_size x 512 x 4 x 4
        self.main.add_module("L1", 
            DCGANDiscLayer("Discriminator_1", opt.nc, opt.ndf, 4, 2, 1, 
                           batch_norm=False))
        
        # ---> 512//2 x 8 x 8
        self.main.add_module("L2", 
            DCGANDiscLayer("Discriminator_2", opt.ndf, opt.ndf*2, 4, 2, 1))
        
        # ---> 512//4 x 16 x 16
        self.main.add_module("L3", 
            DCGANDiscLayer("Discriminator_3", opt.ndf*2, opt.ndf*4, 4, 2, 1))
        
        # ---> 512//8 = 64 x 32 x 32
        self.main.add_module("L4", 
            DCGANDiscLayer("Discriminator_3", opt.ndf*4, opt.ndf*8, 4, 2, 1))
        
        #Now 64 channels to opt.nc x 64 x 64
        self.main.add_module("L5", 
            nn.Conv2d(opt.ndf*8, 1, 4, 1, 0, bias = False))
        
        self.main.add_module("L6", nn.Sigmoid())
    def forward(self, noise):
        return self.main(noise)


# In[ ]:




