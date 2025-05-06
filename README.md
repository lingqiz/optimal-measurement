## Generalized Compressed Sensing for Image Reconstruction with Diffusion Probabilistic Models

Ling-Qi Zhang, Zahra Kadkhodaie, Eero P. Simoncelli, and David H. Brainard   
*Transactions on Machine Learning Research (TMLR)*    
[https://openreview.net/forum?id=lmHh4FmPWZ](https://openreview.net/forum?id=lmHh4FmPWZ)


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

### Examples
See `/notebook` for examples of 
- learning the optimized linear measurement
- running the image reconstruction with the diffusion model
- analyzing the properties of the measurement matrix
