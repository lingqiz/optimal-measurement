import torch, random, math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD
from utils.training import sample_noise
from dataclasses import dataclass

@dataclass
class Args:
    train_noise: list
    test_noise: int = 128
    n_epoch: int = 50
    batch_size: int = 128
    lr: float = 0.05
    decay_lr: float = 0.99
    pow_noise: bool = True
    verbose: bool = False

def sample_noise(size, noise_level, pow_noise):
    noise = torch.empty(size=size)

    for idx in range(int(size[0])):
        # determine noise S.D.
        noise_sd = random.randint(noise_level[0], noise_level[1]) / 255.0

        # biased noise sample (towards smaller noise)
        if pow_noise:
            noise_sd = math.pow(noise_sd, 2)

        # sample Gaussian i.i.d. noise
        noise[idx] = torch.normal(mean=0.0, std=noise_sd, size=list(size[1:]))

    return noise

def train_simple(train_set, test_set, model, args):
    # training with GPU if available
    rank = (0 if torch.cuda.is_available() else 'cpu')
    model = model.to(rank)
    model.train()

    # training dataset
    train_set = DataLoader(train_set, batch_size=args.batch_size,
                shuffle=True, num_workers=4, pin_memory=True)

    # optimizer
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.90)
    scheduler = ExponentialLR(optimizer, gamma=args.decay_lr)
    criterion = nn.MSELoss()

    # run training
    epoch_loss = []
    test_loss = []
    for epoch in range(args.n_epoch):
        model.train()
        total_loss = 0.0

        for count, batch in enumerate(train_set):
            optimizer.zero_grad(set_to_none=True)

            # setup noise and input pair
            batch = batch.to(rank)
            noise = sample_noise(batch.shape, args.train_noise, args.pow_noise).to(rank)
            noise_input = batch + noise

            # forward pass
            residual = model(noise_input)
            loss = criterion(residual, noise)

            # backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # performance on test set
        model.eval()

        test_noise = test_set + torch.normal(0, args.test_noise / 255.0, size=test_set.size())
        with torch.no_grad():
            residual = model(test_noise.to(rank))
            test_denoise = test_noise - residual.cpu()

            test_in = criterion(test_set, test_noise)
            test_out = criterion(test_set, test_denoise)

        # print some diagnostic information
        epoch_loss.append(total_loss / float(count + 1))
        test_loss.append(test_out)

        if args.verbose:
            print('epoch %d/%d' % (epoch + 1, args.n_epoch))
            print('average loss %.4f' % (total_loss / float(count + 1)))
            print('test in %.4f, out %.4f' % (test_in, test_out))

    return epoch_loss, test_loss

def quiver_plot(x, y, model, device, new_fig=True, scale=10):
    # define the mesh grid for the vector field
    X, Y = np.meshgrid(x, y)
    XY = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

    # compute the vector field (log prior gradient)
    XY_tr = torch.from_numpy(XY).float()
    with torch.no_grad():
        UV = - model(XY_tr.to(device)).detach().cpu().numpy()

    U = UV[:, 0].reshape(X.shape)
    V = UV[:, 1].reshape(Y.shape)

    # generate the quiver plot
    if new_fig:
        plt.figure(figsize=(8, 8))

    plt.quiver(X, Y, U, V, scale=scale)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.axis('equal')
    plt.show()