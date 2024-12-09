#%pylab inline
import numpy as np
import random

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import json

from torchvision.transforms.functional import rgb_to_grayscale

from model import Glow
from torch import nn
from torchvision import utils

#one can change 1000 (ie. the number that indicates the training dataset size)
rootdir = "./results/results_glow_model_1000/"
input_folder_base = rootdir + "output_1000_2ndmod_"
input_folder = input_folder_base +"A/"    # model "A"
model_saved =  'model_best_saved.pt'
output_base =  rootdir +"glow_samples_1000_2ndmod_"


with open(input_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)

image_size = hparams['img_size']
n_bits = hparams['n_bits']
n_bins = 2.0 ** n_bits

def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes

def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

groups = ['A','B'] 

def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    return x



temp = 1.0 # temeprature

# Generate at the end of the day 10,000 samples
# if as me you are stuck to to it in one batch, you can proceed in 3 batches as 
# below assocate seed 10, 42, 36 to samples productions 2_000, 4_000 and 4_000
check_manual_seed(36) #was 10 or 42 , 36
n_sample = 2_000 # 4000/2000 

# generate common latent variable
z_sample = []
z_shapes = calc_z_shapes(3, hparams['img_size'], 
                         hparams['n_flow'], hparams['n_block'])
for z in z_shapes:
    z_new = torch.randn(n_sample, *z) * temp
    z_sample.append(z_new.to(device))

all_samples = {}
for group in groups:
    input_folder = input_folder_base + group+"/"
    model_single = Glow(
        3, hparams['n_flow'], hparams['n_block'], affine=hparams['affine'], 
        conv_lu=not hparams['no_lu'])
    
    n_param =  sum(p.numel() for p in model_single.parameters() if p.requires_grad)
    print('Total number of parameters is (million) ' , n_param/1e6)
    
    model = nn.DataParallel(model_single)
    model = model.to(device)
    saved_model = input_folder+model_saved 
    model.load_state_dict(torch.load(saved_model))
    model.eval()
    with torch.no_grad():
        imgs_spl = model_single.reverse(z_sample).detach().cpu().data
        imgs_spl = postprocess(imgs_spl)
        imgs_spl = torch.mean(imgs_spl,axis=1,keepdims=True)    # simple mean = (r+g+b)/3
    all_samples[group] = imgs_spl


fname = fname = output_base +str(n_sample)+"_T"+str(temp)+".pt" 
torch.save(all_samples, fname)



