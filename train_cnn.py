from random import randint
from models.denoiser import Denoiser
from utils.training import train_denoiser, train_parallel
from utils.dataset import DataSet, test_model
from utils.helper import parse_args
import torch, os, torch.multiprocessing as mp

# run argument parser
args = parse_args()

def train(args):
    # load dataset
    print('start loading training data')

    dataset = DataSet.load_dataset(args)
    train_set = torch.from_numpy(dataset.train_set())
    test_set  = torch.from_numpy(dataset.test_set())

    print('dataset loaded, size %d' % train_set.size()[0])

    # train with DDP and Multi-GPUs
    if args.ddp:
        # move dataset to shared memory
        train_set.share_memory_()
        test_set.share_memory_()

        world_size = torch.cuda.device_count()
        print('model training with %d GPUs' % world_size)

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        args.seed = randint(0, 65535)
        mp.set_start_method('spawn')
        mp.spawn(train_parallel, nprocs=world_size,
        args=(world_size, train_set, test_set, args))

    # train with single GPU (or CPUs)
    else:
        # denoiser conv net
        model = Denoiser(args)
        print('number of parameters is ',
            sum(p.numel() for p in model.parameters()))

        print('start training')
        model = train_denoiser(train_set, test_set, model, args)

        # save trained model
        print('save model parameters')
        torch.save(model.state_dict(), args.save_path)

def test(args):
    test_set = DataSet.load_dataset(args, test_mode=True).test_set()

    # load denoiser
    model = Denoiser(args)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    input_psnr = []
    output_psnr = []
    for noise in range(110, 0, -10):
        psnr = test_model(test_set, model, noise, device)[0].mean(axis=1)
        input_psnr.append(psnr[0])
        output_psnr.append(psnr[1])

    print('input psnr: ', ['%.2f' % val for val in input_psnr])
    print('output psnr: ', ['%.2f' % val for val in output_psnr])

    return (input_psnr, output_psnr)

if __name__ == '__main__':
    # only be executed when the script is run directly,
    # and not when it is imported as a module
    # used here together with the multiprocessing module
    print('configuration: \n', args)

    if args.mode == 'train':
        train(args)

    if args.mode == 'test':
        test(args)
