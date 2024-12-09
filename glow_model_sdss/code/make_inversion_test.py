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

import pickle 

###############
# test z~N(0,1) lattent => Glow_gene => x image => Glow_inverse => z'
###############


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


def latent_to_dict(z_list):
    out = {}
    for i in range(len(z_list)):
        out[i] = z_list[i].detach().cpu().numpy()
    return out
    

check_manual_seed(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Change 100000
ntrain = 100_000
rootdir = "./results/results_glow_model_"+str(ntrain) 
input_folder  = rootdir + "output_"+str(ntrain)+"_2ndmod_A/"

model_saved =  'model_best_saved.pt' 

with open(input_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)

image_size = hparams['img_size']


#####
# load model A
#####

model_single_A = Glow(
        3, hparams['n_flow'], hparams['n_block'], affine=hparams['affine'], 
        conv_lu=not hparams['no_lu'])

model_A = nn.DataParallel(model_single_A)
model_A = model_A.to(device)
saved_model = input_folder+model_saved
model_A.load_state_dict(torch.load(saved_model))
model_A.eval()

#####
# load model B
#####

input_folder  = rootdir + "output_"+str(ntrain)+"_2ndmod_B/"


model_single_B = Glow(
        3, hparams['n_flow'], hparams['n_block'], affine=hparams['affine'], 
        conv_lu=not hparams['no_lu'])

model_B = nn.DataParallel(model_single_B)
model_B = model_B.to(device)
saved_model = input_folder+model_saved
model_B.load_state_dict(torch.load(saved_model))
model_B.eval()


##########
temp = 1.0 # temeprature was 0.7
n_sample = 1000 # max 5000
z_sample = []
z_shapes = calc_z_shapes(3, hparams['img_size'], 
                         hparams['n_flow'], hparams['n_block'])

print("z_shapes",z_shapes)

for z in z_shapes:
    z_new = torch.randn(n_sample, *z) * temp
    z_sample.append(z_new.to(device))


with open("./z_init_T"+str(temp)+".pkl",'wb') as f:
    pickle.dump(latent_to_dict(z_sample),f)

with torch.no_grad():
    # x samples from A model
    print("Make image from model A....")
    # generate a sample
    image = model_single_A.reverse(z_sample, reconstruct=False) # use False to get a true sample

    # inversion with model A
    _,_, z_out_A, z_out_norm_A =  model_single_A.forward(image)

    with open("./z_out_norm_A_T"+str(temp)+".pkl",'wb') as f:
        pickle.dump(latent_to_dict(z_out_norm_A),f)

    # inversion with model B
    print("Get back latent from model B....")
    _,_, z_out_B, z_out_norm_B  =  model_single_B.forward(image)
    
    with open("./z_out_norm_B_T"+str(temp)+".pkl",'wb') as f:
        pickle.dump(latent_to_dict(z_out_norm_B),f)


print("Bye...")
