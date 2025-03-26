import argparse
import torch
import numpy as np
from utils.dataset import CelebA, Texture, CIFAR, MNIST, Mixture
from models.unet import init_UNet
from models.unet_flex import load_learned_model
from inverse.lnopt import run_optim, gnl_pca

def args():
    parser = argparse.ArgumentParser(description='Denoiser Training')

    # option/mode for the script
    parser.add_argument('-f',
                        required=False,
                        type=str,
                        help='jupyter notebook')
    parser.add_argument('--mode',
                        required=False,
                        type=str,
                        help='script mode')
    parser.add_argument('--pbar',
                        type=bool,
                        default=False)

    # arguments for optmization
    parser.add_argument('--batch_size',
                        type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=32,
                        help='number of epochs to train')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3)
    parser.add_argument('--decay_rate',
                        type=float,
                        default=0.90)
    parser.add_argument('--loss_type',
                        type=str,
                        default='MSE')
    parser.add_argument('--n_sample',
                        type=int,
                        default=64)
    parser.add_argument('--recon_method',
                        type=str,
                        default='denoiser')

    # see dataset.py for parameters for individual dataset
    parser.add_argument('--data_path',
                        type=str,
                        default='celeba')
    parser.add_argument('--avg',
                        type=bool,
                        default=False)

    # denoiser network
    parser.add_argument('--model_path',
                        type=str,
                        default='unet_celeba')

    # parse arguments and check
    args, _ = parser.parse_known_args()
    return args

# parse arguments
# set file path to denoiser model
args = args()
args.model_path = './assets/' + args.model_path + '.pt'

# load training and test set based on data_path name
if args.data_path == 'celeba':
    data = CelebA(from_numpy=True)
    train_set = data.train_set()

    N_TEST = 256
    test_set = data.test_set()[:N_TEST]

elif args.data_path == 'texture':
    data = Texture(class_name='all')
    train_set = data.train_set()
    test_set = data.test_set()

elif args.data_path == 'cifar':
    data = CIFAR()
    train_set = data.train_set()
    test_set = data.test_set()

elif args.data_path == 'mnist':
    # run on single digit
    DIGIT = None

    data = MNIST()
    if DIGIT is not None:
        data.select_digit(DIGIT)

    train_set = data.train_set()
    test_set = data.test_set()

elif args.data_path == 'mixture':
    data = Mixture()

    train_set = data.train_set()
    test_set = data.test_set()
    model_path = data.model_path

train_set = torch.from_numpy(train_set)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_torch = torch.tensor(test_set).permute([0, 3, 1, 2]).to(device)

print(f'Loaded {args.data_path} dataset')
print(f'Number of training samples: {train_set.shape[0]}')
print(f'Number of test samples: {test_torch.shape[0]}')

# setup save name
save_name = args.data_path
paras = [args.n_sample, args.loss_type, args.batch_size, args.n_epoch,
         args.lr, args.decay_rate, args.pbar, args.avg]

# list all arguments and values
config_str = ' '.join(f'{k}={v}' for k, v in vars(args).items())

if args.recon_method == 'Denoiser':
    # load denoiser model
    # use smaller number of blocks for CIFAR or MNIST dataset
    if args.data_path == 'mixture':
        model = load_learned_model(model_path, print_args=False)
        model = model.eval()

    else:
        if args.data_path == 'cifar' or args.data_path == 'mnist':
            num_blocks = 2
        else:
            num_blocks = 3

        model = init_UNet({'num_blocks':num_blocks})
        model.load_state_dict(torch.load(args.model_path))
        model = model.eval()

    # run optimization
    run_optim(train_set, test_torch, model, save_name, config_str, *paras)

elif args.recon_method == 'Linear':
    # run optimization
    gnl_pca(train_set, test_torch, save_name, config_str, *paras)