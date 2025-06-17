# DP-IQA: Utilizing Diffusion Prior for Blind Image Quality Assessment in the Wild [[Link to paper](https://arxiv.org/abs/2405.19996)]

![Framework](/figures/framework.jpg)

Abstract: *Blind image quality assessment (IQA) in the wild, which assesses the quality of images with complex authentic distortions and no reference images, presents significant challenges. Given the difficulty in collecting large-scale training data, leveraging limited data to develop a model with strong generalization remains an open problem. Motivated by the robust image perception capabilities of pre-trained text-to-image (T2I) diffusion models, we propose a novel IQA method, diffusion priors-based IQA (DP-IQA), to utilize the T2I model's prior for improved performance and generalization ability. Specifically, we utilize pre-trained Stable Diffusion as the backbone, extracting multi-level features from the denoising U-Net guided by prompt embeddings through a tunable text adapter. Simultaneously, an image adapter compensates for information loss introduced by the lossy pre-trained encoder. Unlike T2I models that require full image distribution modeling, our approach targets image quality assessment, which inherently requires fewer parameters. To improve applicability, we distill the knowledge into a lightweight CNN-based student model, significantly reducing parameters while maintaining or even enhancing generalization performance. Experimental results demonstrate that DP-IQA achieves state-of-the-art performance on various in-the-wild datasets, highlighting the superior generalization capability of T2I priors in blind IQA tasks. To our knowledge, DP-IQA is the first method to apply pre-trained diffusion priors in blind IQA.*

# Preparation
We recommend installing 64-bit Python 3.11 and PyTorch 2.6.0. On a CUDA GPU machine, the following will do the trick:

```
pip install -r requirements.txt
```

We have done all testing and development using an A100 GPU.

**Download dataset**

Download the [KonIQ-10k](https://osf.io/hcsdy/) dataset (OSF Storage -> database -> 1024x768). Make sure the path of its .csv file is '/koniq/koniq10k_distributions_sets.csv', and the root path of images is 'koniq/1024x768' in your project.

For zero-shot testing, please download the [CLIVE](https://live.ece.utexas.edu/research/ChallengeDB/index.html) dataset. Make sure the root path of its .mat files is 'clive/ChallengeDB_release/Data', and the root path of images is 'clive/ChallengeDB_release/Images' in your project.

**Download checkpoint**

Put [DP-IQA (student model trained on KonIQ-10k)](https://drive.google.com/file/d/1PNznQU-vuS2ThA6tWT-fy3DmzPIuJRTN/view?usp=drive_link) to '/trained_models'.

The checkpoint provided in the demo is in .onnx format and therefore does not include the code that defines the model. The code of models' definition will be uploaded along with the conventional format (e.g., .pth) checkpoints after the paper is accepted.

# Run this demo
Testing on KonIQ-10k dataset:

```
python koniq_demo.py
```

For zero-shot testing on CLIVE dataset:

```
python koniq_zeroshot_demo.py
```

# Citation

```
@article{fu2024dp,
  title={DP-IQA: Utilizing Diffusion Prior for Blind Image Quality Assessment in the Wild},
  author={Fu, Honghao and Wang, Yufei and Yang, Wenhan and Wen, Bihan},
  journal={arXiv preprint arXiv:2405.19996},
  year={2024}
}
```
