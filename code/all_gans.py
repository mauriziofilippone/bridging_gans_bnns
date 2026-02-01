# Adapted from https://github.com/eriklindernoren/PyTorch-GAN
# Original Copyright (c) 2018 Erik Linder-Norén

import sys
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch

import torchao.quantization as tq

# torch.use_deterministic_algorithms(True)

from sam import SAM
from ema import EMA, enable_bn_running_stats, disable_bn_running_stats

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

os.makedirs("results/", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mnist", help="dataset")
parser.add_argument("--objective", type=str, default="wgan_div", help="GAN objective")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--rho_sam", type=float, default=0.0, help="sam: ball radius")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--n_critic", type=int, default=1, help="how many iterations of optimization of discriminator before optimizing generator")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--mlp_depth", type=int, default=1, help="number of layers in MLP model")
parser.add_argument("--noiselik", type=float, default=0.0, help="sam: ball radius")
parser.add_argument("--lambdagrad", type=float, default=0.0, help="gradient regularization parameter")
parser.add_argument("--p_mcd", type=float, default=0.0, help="dropout probability")
parser.add_argument("--n_mcd", type=int, default=1, help="number of Monte Carlo samples to use at each training iteration")

opt, _ = parser.parse_known_args()
# opt = parser.parse_args()

######################################################################
## Overriding certain hyper-parameters based on dataset and batch-size
## This is to simplify reproducing the results in the paper
if opt.objective == "wgan_div":
    base_lr = 0.001
    opt.n_critic = 5
if opt.objective == "rgan":
    base_lr = 0.0002
    opt.n_critic = 1
if opt.objective == "mmd_gan":
    base_lr = 0.01
    opt.n_critic = 1

######################################################################

SEEDS = (29835, 28347, 192377, 30498, 123817, 16548, 8937, 1827481, 87832, 26564, 65525)
torch.manual_seed(SEEDS[opt.seed])

if opt.dataset == "gauss":
    opt.channels = 2
    opt.n_samples = 200

if opt.dataset == "mnist":
    opt.channels = 1

if opt.dataset == "celeba":
    opt.img_size = 64

if opt.dataset == "ffhq128":
    opt.img_size = 128
    
print(opt)
text_name_run = opt.objective + "_" + opt.dataset + "_seed_" + str(opt.seed) + "_batchsize_" + str(opt.batch_size) + "_noiselik_" + str(opt.noiselik) + "_lambdagrad_" + str(opt.lambdagrad) + "_rho_" + str(opt.rho_sam) + "_p_mcd_" + str(opt.p_mcd) + "_n_mcd_" + str(opt.n_mcd)

cuda = True if torch.cuda.is_available() else False

# Popular initialization function for the weights of the generator and discriminator
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

nc = opt.channels
nz = opt.latent_dim
# number of generator filters
ngf = 64
#number of discriminator filters
ndf = 64

if (opt.dataset=="gauss"):
    from mlp import Generator, Discriminator
    generator = Generator(opt.mlp_depth, 32, nz, opt.channels, opt.p_mcd)
    discriminator = Discriminator(0, 64, nz, opt.channels, opt.p_mcd)
    ema_generator = EMA(generator, decay=0.999)

else:
    if (opt.dataset=="mnist") | (opt.dataset=="cifar10"):
        from dcgan64 import Generator, Discriminator
    if (opt.dataset=="celeba") | (opt.dataset=="ffhq128"):
        from dcgan128 import Generator, Discriminator    

    # Initialize generator and discriminator
    generator = Generator(nz, ngf, ndf, nc, opt.p_mcd)
    discriminator = Discriminator(nz, ngf, ndf, nc, opt.p_mcd)
    ema_generator = EMA(generator, decay=0.999)

if opt.objective == "rgan":
    from objective_rgan import loss_function_G, loss_function_D

if opt.objective == "wgan_div":
    from objective_wgan_div import loss_function_G, loss_function_D

if opt.objective == "mmd_gan":
    from objective_mmd_gan import loss_function_G, loss_function_D
    

if cuda:
    generator.cuda()
    discriminator.cuda()

if(opt.dataset != "gauss"):
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


if opt.dataset == "gauss":
    inputs = torch.randn([opt.n_samples, opt.channels])
    inputs = inputs.detach()
    targets = torch.randn([opt.n_samples])

    inputs  = torch.tensor(inputs)
    targets = torch.tensor(targets)

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=True)


# Configure data loader
if opt.dataset == "mnist":
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST("../../data/mnist", train=True, download=True, transform=transforms.Compose(
                [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),), batch_size=opt.batch_size, shuffle=True,
    )

if opt.dataset == "cifar10":
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root="../../data/cifar10", download=True,
                               transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                             ])), batch_size=opt.batch_size, shuffle=True,
    )

if opt.dataset == "cifar100":
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root="../../data/cifar100", download=True,
                               transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                             ])), batch_size=opt.batch_size, shuffle=True,
    )

if opt.dataset == "celeba":
    dataroot = "../../data/celeba/"
    dataset = datasets.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.img_size * 2),
                                   transforms.CenterCrop(opt.img_size * 2),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

if opt.dataset == "ffhq128":
    dataroot = "../../../images_dataset_ffhq128/"
    dataset = datasets.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.img_size),
                                   transforms.CenterCrop(opt.img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)


# Optimizers
if opt.rho_sam != 0:
    optimizer_G = SAM(generator.parameters(), torch.optim.Adam, rho=opt.rho_sam, adaptive=True, lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = SAM(discriminator.parameters(), torch.optim.Adam, rho=opt.rho_sam, adaptive=True, lr=opt.lr, betas=(opt.b1, opt.b2))
    print("SAM+ADAM OPTIMIZER\n\n")
    optimizer = "sam_adam"

if opt.rho_sam == 0:
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    print("ADAM OPTIMIZER\n\n")
    optimizer = "adam"

lr_scheduler_G = torch.optim.lr_scheduler.LinearLR(optimizer_G, start_factor=0.01, total_iters=10)
lr_scheduler_D = torch.optim.lr_scheduler.LinearLR(optimizer_D, start_factor=0.01, total_iters=10)


# ----------
#  Training
# ----------
generator.train()
discriminator.train()
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        valid = torch.ones((imgs.shape[0]), device=device)
        fake = torch.zeros((imgs.shape[0]), device=device) 
        
        # Configure input
        real_imgs = Variable(imgs, requires_grad=True).to(device) 

        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], opt.latent_dim, 1, 1, device=device) 

        noise = torch.randn(real_imgs.shape, device=device) * opt.noiselik

        # # ---------------------
        # #  Train Discriminator
        # # ---------------------
        if opt.objective == "mmd_gan":
            d_loss = torch.zeros(1)

        else:
            # Generate a batch of images
            fake_imgs = generator(z) + noise
            
            optimizer_D.zero_grad()

            if optimizer == "adam":
                total_d_loss = 0

                for _ in range(opt.n_mcd):
                    d_loss = loss_function_D(fake_imgs, real_imgs, generator, discriminator, fake, valid)
                    total_d_loss += d_loss / opt.n_mcd

                total_d_loss.backward()
                #nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_D.step()

            if optimizer == "sam_adam":
                enable_bn_running_stats(discriminator)
                d_loss1 = loss_function_D(fake_imgs, real_imgs, generator, discriminator, fake, valid)
                d_loss1.backward(retain_graph=True)
                # nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_D.first_step(zero_grad=True)
                disable_bn_running_stats(discriminator)
                d_loss = loss_function_D(fake_imgs, real_imgs, generator, discriminator, fake, valid)
                with torch.autograd.set_detect_anomaly(True):
                    d_loss.backward()
                    # nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    optimizer_D.second_step(zero_grad=True)  
                enable_bn_running_stats(discriminator)

        # -----------------
        #  Train Generator
        # -----------------

        if i % opt.n_critic == 0: 

            optimizer_G.zero_grad()

            # Loss measures generator's ability to fool the discriminator
            if optimizer == "adam":
                total_g_loss = 0
                for _ in range(opt.n_mcd):
                    g_loss = loss_function_G(z, noise, real_imgs, generator, discriminator, fake, valid, opt.lambdagrad)
                    total_g_loss += g_loss / opt.n_mcd

                total_g_loss.backward()
                # nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer_G.step()

            if optimizer == "sam_adam":
                enable_bn_running_stats(generator)
                g_loss1 = loss_function_G(z, noise, real_imgs, generator, discriminator, fake, valid, opt.lambdagrad)
                g_loss1.backward()
                # nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer_G.first_step(zero_grad=True)
                disable_bn_running_stats(generator)
                g_loss = loss_function_G(z, noise, real_imgs, generator, discriminator, fake, valid, opt.lambdagrad)
                with torch.autograd.set_detect_anomaly(True):
                    g_loss.backward()
                    # nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                    optimizer_G.second_step(zero_grad=True)
                enable_bn_running_stats(generator)
                
            if epoch > 20:
                ema_generator.update()


            if (epoch % 10) == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
        
    lr_scheduler_G.step()
    lr_scheduler_D.step()


if (opt.dataset == "gauss"):
    from hessian_eigenthings import compute_hessian_eigenthings
    import matplotlib.pyplot as plt
    
    from scipy.stats import chi2
    quantiles_chi2 = chi2.ppf((0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1), 2)
        
    # def loss_function(z, real_imgs):
    #     return loss_function_G(z, noise, real_imgs, generator, discriminator, fake, valid, opt.lambdagrad)


    # Eigenvalues of the Hessian 
    from objective_mmd_gan import mmd2_rbf_kernel
    
    z = torch.randn(real_imgs.shape[0], opt.latent_dim, 1, 1, device=device) 
    zzset = torch.utils.data.TensorDataset(z, real_imgs)
    zdataloader = DataLoader(zzset, opt.batch_size, shuffle=True)
    num_eigenthings = 1
    use_gpu = True
    if device == 'cpu':
        use_gpu = False
    eigenvals, eigenvecs = compute_hessian_eigenthings(generator, zdataloader, mmd2_rbf_kernel, num_eigenthings, use_gpu=use_gpu)    
    # eig_norm = np.sqrt(np.sum(eigenvals**2))
    # print(eig_norm)
    print(eigenvals)
    filename = "results/top_eigenvalue_" + str(opt.seed) + "_" + opt.dataset + "_" + opt.objective + "_" + str(opt.latent_dim) + "_" + str(opt.mlp_depth) + ".txt"
    np.savetxt(filename, eigenvals, fmt='%.2f')

    filename = "results/final_loss_" + str(opt.seed) + "_" + opt.dataset + "_" + opt.objective + "_" + str(opt.latent_dim) + "_" + str(opt.mlp_depth) + ".txt"
    np.savetxt(filename, (torch.ones([1]) * g_loss).detach().cpu().numpy(), fmt='%.4f')
    
    ## Quantiles
    z = torch.randn(10000, opt.latent_dim, 1, 1, device=device) 
    samples = generator(z)

    square_norm_samples = torch.sum(samples**2, 1)
    
    score = torch.sqrt(torch.mean((torch.histogram(square_norm_samples, torch.Tensor(quantiles_chi2)).hist/1000.0 - 1.0)**2))
    print(score)

    
    ## Plot
    plt.scatter(samples.data.numpy()[:,0], samples.data.numpy()[:,1], c='blue', marker="o")
    plt.scatter(real_imgs.data.numpy()[:,0], real_imgs.data.numpy()[:,1], c='red', marker="x")

    plt.show()
    
    sys.exit(1)


# Save models
os.makedirs("model_saved/", exist_ok=True)
torch.save(generator, "model_saved/generator_" + text_name_run + ".pt")
torch.save(discriminator, "model_saved/discriminator_" + text_name_run + ".pt")

ema_generator.assign()
torch.save(generator, "model_saved/ema_" + text_name_run + ".pt")
discriminator.eval()
generator.eval()

from hessian_eigenthings import compute_hessian_eigenthings

def criterion_wrapper(dummy_z, dummy_real_imgs):

    return loss_function_G(z, noise, real_imgs, generator, discriminator, fake, valid, opt.lambdagrad)

mean_top_eigen = 0
for i, (imgs, _) in enumerate(dataloader):

        valid = torch.ones((imgs.shape[0]), device=device)
        fake = torch.zeros((imgs.shape[0]), device=device) #.squeeze()
        
        # Configure input
        real_imgs = Variable(imgs, requires_grad=True).to(device) # Variable(imgs.type(Tensor))

        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], opt.latent_dim, 1, 1, device=device) 

        noise = torch.randn(real_imgs.shape, device=device) * opt.noiselik

        # z = torch.randn(real_imgs.shape[0], opt.latent_dim, 1, 1, device=device) 
        # zzset = (torch.zeros(opt.batch_size),torch.zeros(opt.batch_size))
        zzset = torch.utils.data.TensorDataset(z, real_imgs)
        # zzset = [(z, real_imgs)]
        zdataloader = DataLoader(zzset, opt.batch_size, shuffle=True)
        num_eigenthings = 1
        use_gpu = True
        if device == 'cpu':
            use_gpu = False
        eigenvals, eigenvecs = compute_hessian_eigenthings(generator, zdataloader, criterion_wrapper, num_eigenthings, full_dataset=True, use_gpu=use_gpu)    
        # eig_norm = np.sqrt(np.sum(eigenvals**2))
        # print(eig_norm)
        print(eigenvals)

        mean_top_eigen += eigenvals / len(dataloader)

print(mean_top_eigen)

filename = "results/top_eigenvalue_" + text_name_run + ".txt"
np.savetxt(filename, mean_top_eigen, fmt='%.2f')

filename = "results/final_loss_" + text_name_run + ".txt"
np.savetxt(filename, (torch.ones([1], device=device) * g_loss).detach().cpu().numpy(), fmt='%.4f')

# Manually set the specific dropout layers back to training mode
generator.dropout1.train()
generator.dropout2.train()
if (opt.dataset=="celeba") | (opt.dataset=="ffhq128"):
    generator.dropout3.train()

os.makedirs("generated_images/", exist_ok=True)
nbatchimgs = 1000
for repetition in range(3):

    dir_name = "generated_images/" + text_name_run + "_rep_" + str(repetition)
    os.makedirs(dir_name, exist_ok=True)
    
    for iii in range(10):
        with torch.no_grad():
            z = torch.randn(nbatchimgs, opt.latent_dim, 1, 1, device=device)
            generated_images = generator(z)
            
        for j, img in enumerate(generated_images):
            img_number = j + iii * nbatchimgs
            save_image(img, f'{dir_name}/image_{img_number}.png')


## Compute the performance metrics on the generated images
torch.cuda.empty_cache()

for repetition in range(3):
    dir_name = "generated_images/" + text_name_run + "_rep_" + str(repetition)

    if opt.dataset != "ffhq128":
        os.system("fidelity --isc  --fid --kid --input1 " + dir_name + " --input2 ../../../images_dataset_" + opt.dataset + "  > results/res_" + text_name_run + "_rep_" + str(repetition))
    if opt.dataset == "ffhq128":
        os.system("fidelity --isc  --fid --kid --input1 " + dir_name + " --input2 ../../../images_dataset_" + opt.dataset + "/subdir" + "  > results/res_" + text_name_run + "_rep_" + str(repetition))

