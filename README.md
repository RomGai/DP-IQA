# DP-IQA: Utilizing Diffusion Prior for Blind Image Quality Assessment in the Wild

![Framework](/figures/framework.jpg)

Abstract: *Image quality assessment (IQA) plays a critical role in selecting high-quality images and guiding compression and enhancement methods in a series of applications. The blind IQA, which assesses the quality of in-the-wild images containing complex authentic distortions without reference images, poses greater challenges. Existing methods are limited to modeling a uniform distribution with local patches and are bothered by the gap between low and high-level visions (caused by widely adopted pre-trained classification networks). In this paper, we propose a novel IQA method called diffusion priors-based IQA (DP-IQA), which leverages the prior knowledge from the pre-trained diffusion model with its excellent powers to bridge semantic gaps in the perception of the visual quality of images. Specifically, we use pre-trained stable diffusion as the backbone, extract multi-level features from the denoising U-Net during the upsampling process at a specified timestep, and decode them to estimate the image quality score. The text and image adapters are adopted to mitigate the domain gap for downstream tasks and correct the information loss caused by the variational autoencoder bottleneck. Finally, we distill the knowledge in the above model into a CNN-based student model, significantly reducing the parameter to enhance applicability, with the student model performing similarly or even better than the teacher model surprisingly. Experimental results demonstrate that our DP-IQA achieves state-of-the-art results on various in-the-wild datasets with better generalization capability, which shows the superiority of our method in global modeling and utilizing the hierarchical feature clues of diffusion for evaluating image quality.*

# Preparation
We recommend installing 64-bit Python 3.8 and [PyTorch 1.12.0](https://pytorch.org/get-started/locally/). On a CUDA GPU machine, the following will do the trick:

```
pip install taming-transformers
pip install clip
pip install ftfy
pip install omegaconf
pip install einops
pip install torchmetrics
pip install pytorch-lightning
pip install kornia
pip install diffusers
pip install accelerate
pip install rawpy
pip install timm
pip install torchvision
pip install transformers==4.31.0
```

The paper is still under review, we are temporarily releasing only a simple demo and sample checkpoints trained on the KonIQ-10k dataset. The complete code and checkpoints are ready and will be uploaded immediately upon acceptance of the paper.

We have done all testing and development using an A100 GPU. But for this demo, a 16G GPU is enough for testing with small batch size.


**Download dataset**

Download the [Koniq-10k](https://github.com/subpic/koniq) dataset. Make sure the path of its csv file is '/koniq/koniq10k_distributions_sets.csv', and the root path of images is 'koniq/1024x768' in your project.

**Download checkpoints**

1. DP-IQA (teacher model):
2. Student model: 
