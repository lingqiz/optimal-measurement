## Generalized Compressed Sensing for Image Reconstruction with Diffusion Probabilistic Models

Ling-Qi Zhang, Zahra Kadkhodaie, Eero P. Simoncelli, and David H. Brainard   
[https://arxiv.org/abs/2405.17456](https://arxiv.org/html/2405.17456v2)

### Installation
```
python -m venv denoiser-recon
source denoiser-recon/bin/activate
pip install -r requirements.txt
```

### Learning image prior and OLMs
```
train_cnn.py: learning the denoiser (diffusion model) prior.
train_olm.py: learning the optimal linear measurement. 
```
