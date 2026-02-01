# Adapted from https://github.com/eriklindernoren/PyTorch-GAN
# Original Copyright (c) 2018 Erik Linder-Norén

import torch

# Loss function 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)

def loss_function_G(z, noise, real_imgs, generator, discriminator, fake, valid, lambdagrad):

    # Generate a batch of images
    gen_imgs = generator(z) + noise
    # Loss measures generator's ability to fool the discriminator
    # Train on fake images
    real_pred = discriminator(real_imgs).detach()
    fake_pred = discriminator(gen_imgs)

    g_loss_1 = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), valid)
    g_loss_2 = adversarial_loss(real_pred - fake_pred.mean(0, keepdim=True), fake)

    g_loss = (g_loss_1 + g_loss_2) / 2
    
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
    real_pred = discriminator(real_imgs)
    # Fake images
    fake_pred = discriminator(fake_imgs.detach())

    real_loss = adversarial_loss(real_pred - fake_pred.mean(0, keepdim=True), valid)
    fake_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), fake)
    
    d_loss = (real_loss + fake_loss) / 2

    return d_loss

