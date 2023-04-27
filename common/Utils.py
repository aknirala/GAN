#!/usr/bin/env python
# coding: utf-8

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




