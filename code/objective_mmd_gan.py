import torch

# Loss function 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gaussian_kernel_naive_implementation(x, y, sigmas):
    """
    Computes the RBF (Gaussian) kernel between two tensors.
    
    Args:
        x (torch.Tensor): First tensor of shape (batch_size, feature_dim).
        y (torch.Tensor): Second tensor of shape (batch_size, feature_dim).
        sigmas (torch.Tensor): A tensor of bandwidths for the kernel.
    
    Returns:
        torch.Tensor: The kernel matrix.
    """
    # Reshape sigmas for broadcasting
    sigmas = sigmas.view(1, 1, -1)
    
    # Expand tensors for element-wise squared difference calculation
    x_expand = x.unsqueeze(1)
    y_expand = y.unsqueeze(0)
    
    # Compute the squared Euclidean distance
    dist = torch.cdist(x, y, p=2.0)**2
    
    # Calculate the kernel matrix for each sigma and sum them up
    kernel = torch.exp(-dist.unsqueeze(2) / (2 * sigmas**2)).sum(dim=2)
    
    return kernel


def gaussian_kernel(x, y, lambdas):
    """
    Computes the RBF (Gaussian) kernel between two tensors.
    
    Args:
        x (torch.Tensor): First tensor of shape (batch_size, feature_dim).
        y (torch.Tensor): Second tensor of shape (batch_size, feature_dim).
        lambdas (torch.Tensor): A tensor of inverse bandwidths for the kernel.
    
    Returns:
        torch.Tensor: The kernel matrix.
    """
    # Reshape lambdas for broadcasting
    lambdas = lambdas.view(1, 1, -1)

    tmp1 = torch.sum(x**2, 1)
    tmp2 = torch.sum(y**2, 1)
    tmp3 = torch.matmul(x, y.t())
  
    # Compute the squared Euclidean distance
    dist2 = tmp1.reshape(-1,1) + tmp2 - 2 * tmp3
    
    # Calculate the kernel matrix for each sigma and sum them up
    kernel = torch.exp(-dist2.unsqueeze(2) * lambdas).sum(dim=2)
    
    return kernel


def mmd2_rbf_kernel(x, y, lambdas=None):
    """
    Calculates the Maximum Mean Discrepancy (MMD) between two sets of samples
    using a sum of RBF kernels with different bandwidths.
    
    Args:
        x (torch.Tensor): Samples from the first distribution, shape (n_samples, feature_dim).
        y (torch.Tensor): Samples from the second distribution, shape (n_samples, feature_dim).
        lambdas (list or torch.Tensor): List of inverse bandwidths for the RBF kernels.
                                       If None, a default set of lambdas will be used.
    
    Returns:
        torch.Tensor: The MMD value.
    """
    if lambdas is None:
        lambdas = torch.Tensor([1, 0.5, 0.1, 0.02, 0.01, 0.002, 0.001])

    # Calculate the kernel matrices
    K_xx = gaussian_kernel(x, x, lambdas)
    K_yy = gaussian_kernel(y, y, lambdas)
    K_xy = gaussian_kernel(x, y, lambdas)

    # The MMD squared is given by:
    # mmd^2 = (1/n^2) * sum(K(x_i, x_j)) + (1/m^2) * sum(K(y_i, y_j)) - (2/nm) * sum(K(x_i, y_j))
    # where n and m are the number of samples.
    # In this case, we assume n=m for simplicity.
    mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    
    # return torch.sqrt(mmd2.clamp(min=1-6))
    return mmd2

def loss_function_G(z, noise, real_imgs, generator, discriminator, fake, valid, lambdagrad):

    # Generate a batch of images
    gen_imgs = generator(z) + noise

    gen_imgs.requires_grad_(True)
    
    g_loss = mmd2_rbf_kernel(real_imgs, gen_imgs)

    gradients = torch.autograd.grad(
        outputs=g_loss,
        inputs=gen_imgs,
        grad_outputs=torch.ones(g_loss.size(), device=g_loss.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm_squared = gradients.norm(2, dim=1).mean()
    
    total_loss = g_loss + lambdagrad * grad_norm_squared
    
    return total_loss


def loss_function_D(fake_imgs, real_imgs, generator, discriminator, fake, valid):
    """
    The loss for the discriminator does not need to be computed - in MMD the discriminator is optimized in closed form
    We return a zero
    """

    d_loss = 0.0

    return d_loss


