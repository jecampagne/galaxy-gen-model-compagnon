import numpy as np
import os

import torch

from network import UNet
from model_loader_func import load_UNet
from plotting_func import plot_many_denoised,  show_im_set, plot_single_im
from quality_metrics_func import calc_psnr, im_set_corr
from linear_approx import calc_jacobian, traj_projections
from inverse_tasks_func import synthesis
from algorithm_inv_prob import univ_inv_sol
from dataloader_func import add_noise_torch

print(torch.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

## Load pretraiend denoisers 

denoisers_sdss = {}
# denoisers in group A are trained on one partision of the data which is non-overlapping with partision B
groups = ['A']   #['A', 'B'] 
swap = False
training_data_name = 'sdss'

Ns = [10000,1000] #size of the dataset


for group in groups: 
    print('loading group ' , group )
    denoisers_sdss[group] = {}
    if group == 'B': 
        swap = True
    for N in Ns:       
        try: 
            denoisers_sdss[group][N] = load_UNet(
                           base_path = '../denoisers/UNet',
                           training_data_name= training_data_name, 
                           training_noise='0to255',
                           RF=90,
                           set_size=N, 
                           swap=swap);
        except FileNotFoundError: 
            pass 


# laod dataset

train_sdss ={}
test_sdss = {}
train_sdss['A'] = {}
train_sdss['B'] ={}
test_sdss['B'] = {}
test_sdss['A'] = {}

data = torch.load('../../datasets/sdss_train_no_repeats_64x64.pt')
K = data.shape[2]
print(data.shape)
for N in Ns:
    train_sdss['A'][N] = data[0:N]
    test_sdss['A'][N] = data[-N::]
    train_sdss['B'][N] = data[-N::]
    test_sdss['B'][N] = data[0:N]   

# generate samples starting from all_interm_Ysme seed using both denoisers 
synth = synthesis() 


for N in Ns:
    print(">>> gp",group)        

    n_samples = 10000
    shift = 0
    seeds = range(shift,shift+n_samples)
    freq = 100

    all_samples = {}
    for group in groups:
        print('--------- N : ', N)        

        all_samples[group] = torch.zeros(n_samples, 1, K,K).to(device)
        
        for n in range(n_samples): 
            torch.manual_seed(seeds[n])
            sample, _,_,_ = univ_inv_sol(denoisers_sdss[group][N],
                                         x_c= torch.zeros(1,K,K).cuda(),
                                         task=synth,
                                         device=device,
                                         sig_0=1, 
                                         sig_L=.1, 
                                         h0=.01 , 
                                         beta=.1 , 
                                         freq=freq,
                                         seed = seeds[n], 
                                         init_im = train_sdss[group][N].mean(dim=0).cuda() + torch.randn(1,K,K, device = device), 
                                         init_noise_mean=0,
                                         max_T=10000, 
                                         fixed_h=False)
            all_samples[group][n, 0] = sample.detach()
        # end loop on samples
    # end loop on groups

    # save
    fname = '../results/img_align_sdss_64x64/UNet_many_samples_'+str(N)+"_"+str(n_samples) +'.pt'
    torch.save(all_samples, fname)
    print("done ",fname)

print("All done")



