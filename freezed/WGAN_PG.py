#!/usr/bin/env python
# coding: utf-8

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

