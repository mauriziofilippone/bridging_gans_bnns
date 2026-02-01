# Code to reproduce the results in the paper: Bridging GANs and Bayesian Neural Networks via Partial Stochasticity - https://arxiv.org/abs/2507.00651

Before running the code, you should ensure that you have access to the data used in this work.

MNIST and CIFAR10 are downloaded automatically and placed in the ../../data/ directory.
FFHQ and CELEBA need to be downloaded from https://github.com/NVlabs/ffhq-dataset and https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.

We increased the size of MNIST and CIFAR10 images to 64x64, and we increased the size of CELEBA images to 128x128. 
We reduced the size of the original FFHQ images to 128x128x3.

In order to compute the performance metrics, you need to save the images from the data sets above as PNG and place them in the following directories:
../../../images_dataset_mnist
../../../images_dataset_cifar10
../../../images_dataset_ffhq128
../../../images_dataset_celeba

For FFHQ128 we use the same directory ../../../images_dataset_ffhq128 to construct the data loader.

In the directory code/run_examples.txt you can find some examples on how to run the code.

## Acknowledgments & Attribution
This repository is a hybrid implementation based on the following open-source projects:

* **[PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)** by Erik Linder-Norén (MIT License): Used for the core training framework and GAN objectives.
* **[SAM](https://github.com/davda54/sam)** by David Samuel (MIT License): Integrated the `sam.py` optimizer.
* **[DCGAN-PyTorch](https://github.com/Lornatang/DCGAN-PyTorch)** by Lornatang (Apache License 2.0): Provided the base DCGAN architecture.

### Modifications
In accordance with the Apache License 2.0, please note the following significant changes to the original `DCGAN-PyTorch` code:
- Added **Dropout layers** to the Generator and Discriminator architectures to test Monte Carlo Dropout.
- Integrated the model with the SAM optimizer and regularization techniques with the PyTorch-GAN training loop.

## License
My original contributions are licensed under the **MIT License**. However, this repository contains code from third parties licensed under **MIT** and **Apache License 2.0**. See the LICENSE file for the full text of these licenses.
