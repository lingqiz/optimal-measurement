import torch
import numpy as np
import torch.nn as nn
import logging
import sys
from tqdm import tqdm
from inverse.orthogonal import LinearInverse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import StructuralSimilarityIndexMeasure
from sklearn.decomposition import PCA
from skimage.metrics import peak_signal_noise_ratio as psnr

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MSE = nn.MSELoss(reduction='sum').to(DEVICE)
SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

def pca_projection(train_set, test_torch, n_sample, im_size):
    # Compute PCA on the training set
    image_vec = train_set.permute([0, 3, 2, 1]).flatten(1).numpy()
    pca = PCA(n_components=n_sample).fit(image_vec)

    # PCA reconstruction on test set
    mtx = torch.from_numpy(pca.components_).to(DEVICE)
    image_vec = test_torch.transpose(2, 3).flatten(1)
    recon_vec = image_vec @ mtx.t() @ mtx
    recon_torch = recon_vec.reshape([-1, *im_size]).transpose(2, 3)

    # compute metric
    mse_val = MSE(test_torch, recon_torch) / test_torch.shape[0]
    ssim_val = SSIM(recon_torch, test_torch)
    psnr_val = psnr(image_vec.detach().cpu().numpy(),
                    recon_vec.detach().cpu().numpy())

    # reconstructed test images
    recon_numpy = recon_torch.permute([0, 2, 3, 1]).detach().cpu().numpy()
    return mtx, mse_val.item(), ssim_val.item(), psnr_val, recon_numpy

def denoiser_avg(test_torch, solver, n_avg=5):
    with torch.no_grad():
        # compute average reconstruction
        image_sum = torch.zeros_like(test_torch)
        for _ in range(n_avg):
            image_sum += solver(test_torch)
        recon = image_sum / n_avg

        # compute metric
        mse_val = MSE(test_torch, recon) / test_torch.shape[0]
        ssim_val = SSIM(recon, test_torch)

        test_numpy = test_torch.permute([0, 2, 3, 1]).detach().cpu().numpy()
        recon_numpy = recon.permute([0, 2, 3, 1]).detach().cpu().numpy()
        psnr_val = psnr(test_numpy, recon_numpy)

        return mse_val.item(), ssim_val.item(), psnr_val, recon_numpy

def ln_optim(solver, loss, train, test,
             batch_size=200, n_epoch=50,
             lr=1e-3, gamma=0.95):

    # training data
    n_batch = np.ceil(train.shape[0] / batch_size)
    train_data = DataLoader(train, batch_size, shuffle=True,
                            num_workers=8, pin_memory=True)
    # optimizers
    optim = torch.optim.Adam(solver.parameters(), lr=lr)
    scheduler = ExponentialLR(optim, gamma=gamma, verbose=True)

    batch_loss = []
    epoch_loss = []
    pbar = tqdm(total=int(n_batch))

    # Run n_epoch of training
    for epoch in range(n_epoch):
        pbar.reset()
        total_loss = 0.0

        # SGD Optimization
        for count, batch in enumerate(train_data):
            optim.zero_grad(set_to_none=True)
            batch = batch.permute(0, 3, 1, 2).contiguous().to(DEVICE)

            # run reconstruction
            recon = solver(batch)
            error = loss(batch, recon)

            # optim step
            error.backward()
            optim.step()

            # record loss value
            pbar.update()
            loss_val = error.item() / batch.shape[0]
            batch_loss.append(loss_val)
            total_loss += loss_val

        # average loss value per batch
        avg_loss = total_loss / float(count)
        epoch_loss.append(avg_loss)

        # adjust learning rate
        scheduler.step()

        # compute performance on test set
        mse_val, ssim_val, psnr_val = denoiser_avg(test, solver)[:-1]

        # log training information
        logging.info('Epoch %d/%d' % (epoch + 1, n_epoch))
        logging.info('Training loss value %.3f' % (avg_loss))
        logging.info('Test MSE %.3f, SSIM %.3f, PSNR %.3f \n' % \
                                    (mse_val, ssim_val, psnr_val))
    return solver

def run_optim(train_set, test_torch, denoiser, n_sample, loss='MSE',
                batch_size=200, n_epoch=75, lr=1e-3, gamma=0.95):

    # print relevant information
    run_name = './design/results/%d_%s_im48' % (n_sample, loss)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt='%m/%d/%Y %I:%M:%S %p',
        handlers=[
            logging.FileHandler(run_name + '.log'),
            logging.StreamHandler(sys.stdout)])

    # image and dataset size
    im_size = test_torch.size()[1:]
    logging.info('# Training Data: %d' % train_set.shape[0])
    logging.info('# Test Data: %d \n' % test_torch.shape[0])

    # wrap the model in DataParallel
    solver = LinearInverse(n_sample, im_size, denoiser).to(DEVICE)
    solver.max_t = 60
    solver_gpu = torch.nn.DataParallel(solver)

    # set up loss function for running the optimization
    if loss == 'MSE':
        loss = nn.MSELoss(reduction='sum').to(DEVICE)

    elif loss == 'SSIM':
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='sum').to(DEVICE)
        loss = lambda pred, target: 1.0 - ssim(pred, target)

    # test with PCA for baseline performance
    pca_mtx, mse_val, ssim_val, psnr_val, pca_recon = \
            pca_projection(train_set, test_torch, n_sample, im_size)
    logging.info('PCA MSE %.3f, SSIM %.3f, PSNR %.3f \n' % \
                                (mse_val, ssim_val, psnr_val))

    # denoiser reconstruction with PCA matrix
    solver_pca = LinearInverse(n_sample, im_size, denoiser).to(DEVICE).assign(pca_mtx)
    mse_val, ssim_val, psnr_val, denoiser_recon = denoiser_avg(test_torch, solver_pca)
    logging.info('Denoiser-PCA MSE %.3f, SSIM %.3f, PSNR %.3f \n' % \
                                        (mse_val, ssim_val, psnr_val))

    # run optimization
    solver_optim = ln_optim(solver_gpu, loss, train_set, test_torch,
            batch_size=batch_size, n_epoch=n_epoch, lr=lr, gamma=gamma)
    mse_val, ssim_val, psnr_val, denoiser_optim = denoiser_avg(test_torch, solver_optim)    

    # save results
    pca_mtx = pca_mtx.detach().cpu().numpy()
    optim_mtx = solver.linear.weight.detach().cpu().numpy()
    save_vars = [pca_recon, denoiser_recon, denoiser_optim, pca_mtx, optim_mtx]
    with open(run_name + '.npy', 'wb') as fl:
        [np.save(fl, var) for var in save_vars]

    return save_vars
