from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
import os
import json
import shutil
import random
import csv

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))

    
def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.CenterCrop(image_size),  # should start by PIL Image (H x W x C) in the range [0, 255]
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),   # transform C H W in the range [0.,1.]
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=6) # was 4
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=6) # was 4
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(args, model, optimizer):

    loss_fn = "loss_history_"+args.group+".csv"
    
    if not os.path.isfile(loss_fn):
        file1 = open(loss_fn, "w")
        file1.close()

    
    dataset = iter(sample_data(args.dataroot, args.batch, args.img_size))

    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))


    best_loss = args.best_loss   # JEC 7oct24
    
    with tqdm(range(args.iter_start, args.iter), disable=True) as pbar:
        for i in pbar:
            image, _ = next(dataset)

            image = image.to(device)

            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    #JEC 16nov24 the model forward output has a new item (to be ignored here)
                    log_p, logdet, *_ = model.module(
                        image + torch.rand_like(image) / n_bins
                    )

                    continue

            else:
                #JEC 16nov24 the model forward output has a new item (to be ignored here)
                log_p, logdet, *_ = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()


            ##### JEC 5May24 pb loss=NaN CelebA => clipping
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            if i % 100 == 0:
                print(f"i:{i} Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}")
                with open(loss_fn, "a", newline='') as f2:
                    writer = csv.writer(f2,delimiter=' ')
                    writer.writerow([f"{i}",f"{loss.item():.5f}"])
                    

            if i % 100 == 0 and args.sampling:
                with torch.no_grad():
                    sample_fn = "sample_"+args.group+f"/{str(i + 1).zfill(6)}.png"
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        sample_fn,
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            #save model & optimizer 
            if i % 100 == 0:
                torch.save(
                    model.state_dict(),args.output_dir+"/model_saved.pt"
                )
                torch.save(
                    optimizer.state_dict(), args.output_dir+"/optim_saved.pt"
                )

            if loss.item() < best_loss:
                print(f"i:{i} BEST Loss: {loss.item():.5f}")
                best_loss = loss.item()
                torch.save(
                    model.state_dict(),args.output_dir+"/model_best_saved.pt"
                )
                torch.save(
                    optimizer.state_dict(), args.output_dir+"/optim_best_saved.pt"
                )
                


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Glow trainer")

    
    parser.add_argument(
        "--saved_model",
        default="",
        help="Path to model to load for continuing training",
    )

    parser.add_argument(
        "--saved_optimizer",
        default="",
        help="Path to optimizer to load for continuing training",
    )

    parser.add_argument("--batch", default=16, type=int, help="batch size")

    parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")

    parser.add_argument("--iter_start", default=0, type=int, help="first iteration")

    parser.add_argument("--n_flow", default=32, type=int,
                        help="number of flows in each block"
    )
    parser.add_argument("--n_block", default=4, type=int, help="number of blocks")

    parser.add_argument(
        "--no_lu",
        action="store_true",
        help="use plain convolution instead of LU decomposed version",
    )

    parser.add_argument(
        "--affine", action="store_true", help="use affine coupling instead of additive"
    )

    parser.add_argument("--n_bits", default=5, type=int, help="number of bits")

    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")

    parser.add_argument("--best_loss", default=float('inf'), type=float, help="best loss at begin")
    
    parser.add_argument("--img_size", default=64, type=int, help="image size")

    parser.add_argument("--temp", default=1.0, type=float, help="temperature of sampling")

    parser.add_argument("--n_sample", default=20, type=int, help="number of samples")

    parser.add_argument("--dataroot", type=str, help="Path to image directory")

    parser.add_argument(
        "--output_dir",
        default="output",
        help="Directory without / at end to output logs and model/optim states (will add the group tag)",
    )

    parser.add_argument("--group", type=str, default="def", help="give a tag to output dir, loss file...")

    parser.add_argument(
        "--fresh", action="store_true", help="Remove output directory before starting"
    )

    parser.add_argument("--seed", type=int, default=0, help="manual seed")


    parser.add_argument(
        "--sampling", action="store_true", help="sample images during training")

    
    args = parser.parse_args()


    args.dataroot   = args.dataroot + args.group
    args.output_dir = args.output_dir+ "_"+ args.group + "/" 
    
    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        if args.fresh:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        if (not os.path.isdir(args.output_dir)) or (
            len(os.listdir(args.output_dir)) > 0
        ):
            raise FileExistsError(
                "Please provide a path to a non-existing or empty directory. Alternatively, pass the --fresh flag."  # noqa
            )

    kwargs = vars(args)
    del kwargs["fresh"]


    check_manual_seed(args.seed)
        
    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )

    model = nn.DataParallel(model_single)
    model = model.to(device)
    if args.saved_model:
        model.load_state_dict(torch.load(args.saved_model))
        #file_name, ext = os.path.splitext(args.saved_model)
        #args.iter_start= int(file_name.split("_")[-1])
        #####info_dict = torch.load(os.path.dirname(args.saved_model)+'/optim_saved.pt')
        info_dict = torch.load(args.saved_optimizer)
        iter_start = info_dict['state'][0]['step']+1
        args.iter_start = int(iter_start.item())
        print(f"JEC iter_start: {args.iter_start}")

        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.saved_optimizer:
        optimizer.load_state_dict(torch.load(args.saved_optimizer))

    #save args state
    print("dbg:",kwargs)
    with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)
    
    train(args, model, optimizer)
