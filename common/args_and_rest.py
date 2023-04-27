#!/usr/bin/env python
# coding: utf-8

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

