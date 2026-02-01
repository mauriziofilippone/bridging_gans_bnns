# Adapted from https://github.com/eriklindernoren/PyTorch-GAN
# Original Copyright (c) 2018 Erik Linder-Norén

import torch
from torch.autograd import Variable
import torch.autograd as autograd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# Loss function
adversarial_loss = torch.nn.BCELoss().to(device)

k = 2
p = 6
    
def loss_function_G(z, noise, real_imgs, generator, discriminator, fake, valid, lambdagrad):

    # Generate a batch of images
    fake_imgs = generator(z) + noise
    # Loss measures generator's ability to fool the discriminator
    # Train on fake images
    fake_validity = discriminator(fake_imgs)
    g_loss = -torch.mean(fake_validity)

    grad_norm_squared = 0.0
    if lambdagrad > 0.0:
        g_loss.backward(retain_graph=True)
        for param in generator.parameters():
            if param.grad is not None:
                # We are interested in the L2 norm of the gradients with respect to parameters                                                                                                 
                # param.grad already holds these gradients after main_loss.backward()                                                                                                          
                grad_norm_squared += torch.norm(param.grad) ** 2 # Sum of squared L2 norms                                                                                                     

    return g_loss + lambdagrad * grad_norm_squared

def loss_function_D(fake_imgs, real_imgs, generator, discriminator, fake, valid):

    # Real images
    real_validity = discriminator(real_imgs)
    # Fake images
    fake_validity = discriminator(fake_imgs)

    # Compute W-div gradient penalty
    real_grad_out = Variable(Tensor(real_imgs.size(0)).fill_(1.0), requires_grad=False)
    fake_grad_out = Variable(Tensor(fake_imgs.size(0)).fill_(1.0), requires_grad=False)

    real_grad = autograd.grad(
        real_validity, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
    
    fake_grad = autograd.grad(
        fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
    
    div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
    
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

    return d_loss


def loss_function_D_no_reg(fake_imgs, real_imgs):

    # Real images
    real_validity = discriminator(real_imgs)
    # Fake images
    fake_validity = discriminator(fake_imgs)

    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

    return d_loss
