from inverse.solver import RenderMatrix
from torch.linalg import vector_norm as vnorm
import torch.nn.utils.parametrizations as para
import torch, numpy as np
import torch.nn as nn
import scipy.linalg

def batch_recon(x, solver, batch_size=32):
    '''
    Split input into multiple batches and run reconstruction
    '''
    # split input into batches
    x_split = torch.split(x, batch_size, dim=0)

    for i, x_batch in enumerate(x_split):
        # run reconstruction
        with torch.no_grad():
            recon_batch = solver(x_batch)

        # save results
        if i == 0:
            recon = recon_batch
        else:
            recon = torch.cat([recon, recon_batch], dim=0)

    return recon

class OrthMatrix(RenderMatrix):
    """
    Define a measurement matrix that is orthgonal
    using householder parameterization
    """
    def __init__(self, n_sample, im_size, device):
        n_pixel = np.prod(im_size)

        # init orthgonal matrix with householder product parameterization
        linear = torch.nn.Linear(n_pixel, n_sample).to(device)
        self.para = para.orthogonal(linear, orthogonal_map='householder')

        super().__init__(self.para.weight, im_size, device)

    def forward(self):
        # generate the measurement matrix based on the parameterization
        self.R = self.para.weight

class LinearInverse(nn.Module):
    """
    Implement the linear inverse reconstruction produce as a nn.Module object,
    with the sample - recon calculation as the forward function.

    The measurement matrix is part of the model, and is assumed to be orthogonal.

    This enables us to perform reconstruction on mini-batch and multi-GPU training.
    """

    def __init__(self, n_sample, im_size, denoiser, init_im=None):
        super().__init__()

        # save variables
        self.model = denoiser
        self.im_size = im_size
        self.n_pixel = np.prod(im_size)
        self.n_sample = n_sample
        self.h_increase = True

        # no grad flag for the denoiser model
        for param in self.model.parameters():
            param.requires_grad = False

        # initialize an orthogonal linear measurement matrix
        linear = torch.nn.Linear(self.n_pixel, self.n_sample)
        self.linear = para.orthogonal(linear, orthogonal_map='householder')
        self.mtx = self.linear.weight

        # default parameter for the reconstruction
        self.h_init = 0.10
        self.beta = 0.20
        self.sig_end = 0.01
        self.max_t = 100
        self.last_t = None
        self.run_avg = False
        self.num_avg = 2

        # initialization image
        if init_im is None:
            # default to the grand mean of around 0.45
            init_im = 0.45 * torch.ones([*self.im_size])

        self.init_im = nn.Parameter(init_im.unsqueeze(0), requires_grad=False)

    def refresh(self):
        self.mtx = self.linear.weight

    def assign(self, mtx):
        """
        Assign a measurement matrix. If the matrix is not orthogonal,
        its orthogonal component will be obtained using QR decomposition.
        """
        self.linear.weight = mtx
        self.refresh()

        return self

    def to(self, device):
        return_val = super().to(device)
        self.refresh()
        return return_val

    def log_grad(self, x):
        return - self.model(x)

    def measure(self, x):
        """
        x: images of size [N, C, W, H]

        Note a transpose is required due to different
        in convention between MATLAB and torch
        (Row-Major vs Column-Major), and here we are
        using the MATLAB convention for compatibility
        """
        return torch.matmul(x.transpose(2, 3).flatten(start_dim=1), self.mtx.t())

    def recon(self, m):
        """
        m: vectors of size [N, M]
        """
        new_shape = [-1, self.im_size[0], self.im_size[2], self.im_size[1]]
        return torch.matmul(m, self.mtx).reshape(new_shape).transpose(2, 3)

    def inverse(self, msmt):
        """
        msmt: measurements of images

        Perform denoiser reconstruction on
        images based on linear measurements
        """
        # measurement matrix calculation
        M = self.measure
        M_T = self.recon

        # init variables
        proj = M_T(msmt)
        e = self.init_im.repeat([proj.shape[0], 1, 1, 1])
        n = torch.numel(e[0])
        mu = (e - M_T(M(e))) + proj
        y = torch.randn_like(mu) + mu
        sigma = vnorm(self.log_grad(y),
                dim=(1, 2, 3)) / np.sqrt(n)

        # crose-to-fine reconstruction
        t = 1
        # stop criteria (mean) for batch input
        while torch.mean(sigma) > self.sig_end:
            # update step size
            h = self.h_init
            if self.h_increase:
                h = (self.h_init * t) / (1 + self.h_init * (t - 1))

            # projected log prior gradient
            d = self.log_grad(y)
            d = (d - M_T(M(d)) + proj - M_T(M(y)))

            # noise magnitude
            sigma = vnorm(d, dim=(1, 2, 3)) / np.sqrt(n)

            # injected noise
            gamma = np.sqrt((1 - self.beta * h) ** 2 - (1 - h) ** 2) * sigma
            noise = torch.randn_like(y)
            # expand gamma for shape matching
            gamma = gamma[:, None, None, None].repeat([1, *self.im_size])

            # update image
            y = y + h * d + gamma * noise
            t += 1

            # safe guard for iteration limit (GPU memory limit)
            # (typically) use in conjection with grad=True
            if t > self.max_t:
                break

        self.last_t = t

        # run a final denoise step and return the results
        return y + self.log_grad(y)

    def _run_recon(self, x):
        """
        x: images of size [N, C, W, H]

        Perform denoiser reconstruction on
        images based on linear measurements
        """
        # update the measurement matrix
        # based on the parameterization
        self.refresh()

        # compute the linear measurement
        msmt = self.measure(x)

        # run the reconstruction routine
        return self.inverse(msmt)

    # average over multiple runs
    def average(self, x, num_avg=2):
        # make stacked copy of x
        x_stack = x.repeat([num_avg, 1, 1, 1])

        # run reconstruction
        recon = self._run_recon(x_stack)

        # average over the stack
        recon_avg = recon.reshape([num_avg, -1, *self.im_size])
        recon_avg = torch.mean(recon_avg, dim=0)

        return recon_avg

    def forward(self, x):
        # average over multiple runs
        if self.run_avg:
            return self.average(x, num_avg=self.num_avg)

        # return single sample
        return self._run_recon(x)

class LinearProjection(nn.Module):
    '''
    Linear projection with orthogonalized measurement matrix
    '''

    def __init__(self, n_sample, im_size):
        super().__init__()

        self.im_size = im_size
        self.n_pixel = np.prod(im_size)
        self.n_sample = n_sample

        # initialize an orthogonal linear measurement matrix
        linear = torch.nn.Linear(self.n_pixel, self.n_sample)
        self.linear = para.orthogonal(linear, orthogonal_map='householder')

    def refresh(self):
        self.mtx = self.linear.weight

    def forward(self, x):
        # measurement matrix
        self.refresh()
        proj_mtx = self.mtx

        # compute projection
        x_flat = x.transpose(2, 3).flatten(1)
        recon_flat = x_flat @ proj_mtx.t() @ proj_mtx

        new_shape = [-1, self.im_size[0], self.im_size[2], self.im_size[1]]
        return recon_flat.reshape(new_shape).transpose(2, 3)

class Sequential():
    def init_mtx(self, msmt_mtx):
        '''
        Initialize with a given measurement matrix of shape [k, n]
        '''
        # compute null space of shape [n, n - k]
        null_space = scipy.linalg.null_space(msmt_mtx).T
        self.null_space = torch.nn.Parameter(torch.from_numpy(null_space).float(),
                                             requires_grad=False)
        self.mtx_base = torch.nn.Parameter(torch.from_numpy(msmt_mtx).float(),
                                           requires_grad=False)

        # parameterization
        vector = torch.nn.Linear(self.null_space.shape[0], 1)
        self.vector = para.orthogonal(vector, orthogonal_map='householder')

        # compute measurement matrix
        self.refresh()

    def refresh(self):
        # compute measurement matrix
        vector = self.vector.weight
        msmt_new = vector @ self.null_space
        self.mtx = torch.cat([self.mtx_base, msmt_new], dim=0)

class LinearSequential(Sequential, LinearProjection):
    def __init__(self, n_sample, im_size, msmt_mtx):
        '''
        Initialize with a given measurement matrix of shape [k, n]
        '''
        LinearProjection.__init__(self, n_sample, im_size)
        self.linear = None

        # init parameterization
        self.init_mtx(msmt_mtx)

class InverseSequential(Sequential, LinearInverse):
    def __init__(self, n_sample, im_size, denoiser, msmt_mtx, init_im=None):
        LinearInverse.__init__(self, n_sample, im_size, denoiser, init_im)

        self.linear = None
        self.mtx = None

        # init parameterization
        self.init_mtx(msmt_mtx)

    def assign(self, _):
        # not implemented for the sequential parameterization
        raise NotImplementedError

class GaussianLinearInverse(LinearInverse):
    def __init__(self, n_sample, im_size, denoiser, init_im=None):
        '''
        Linear Inverse with Gaussian measurement noise
        '''
        super().__init__(n_sample, im_size, denoiser, init_im)
        self.noise_sd = 0.01

    def _run_recon(self, x):
        """
        Override the reconstruction function to add Gaussian noise
        """
        # update the measurement matrix
        # based on the parameterization
        self.refresh()

        # compute the linear measurement
        msmt = self.measure(x)

        # add Gaussian noise
        noise = self.noise_sd * torch.rand_like(msmt[0]).reshape([1, -1])
        msmt = msmt + noise.repeat([msmt.shape[0], 1])

        # run the reconstruction routine
        return self.inverse(msmt)