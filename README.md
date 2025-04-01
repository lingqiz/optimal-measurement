## Optimized Linear Measurements for Image Reconstruction

**Optimal compressed sensing for image reconstruction with diffusion probabilistic models**    
Ling-Qi Zhang, Zahra Kadkhodaie, Eero P. Simoncelli, and David H. Brainard   
[https://arxiv.org/abs/2405.17456](https://arxiv.org/html/2405.17456v2)

### Installation
```
python -m venv denoiser-recon
source denoiser-recon/bin/activate
pip install -r requirements.txt
```

### Learning Image Prior and OLMs
```
train_cnn.py: learning the denoiser (diffusion model) prior.
train_olm.py: learning the optimal linear measurement. 
```
