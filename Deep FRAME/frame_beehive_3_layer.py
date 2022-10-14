import os
import random
import logging
import sys
import shutil
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils

def opts(output_dir):
    args = argparse.Namespace()

    args.gpu = 1
    args.num_epoch = 2001
    args.img_size = 224

    args.data_path = './data/beehive'
    args.output_dir = '{}/images'.format(output_dir)
    args.log_dir = '{}/log'.format(output_dir)
    args.ckpt_dir = '{}/ckpt'.format(output_dir)

    args.num_chains = 1
    args.sigma = 1.0
    args.langevin_step_num = 10
    # TODO tune step size
    args.langevin_step_size = 0.9
    # TODO tune learning rate
    args.lr = [0.005, 0.01, 0.00001, 0.03, 0.005, 0.0003]
    args.beta1 = 0.5

    args.debug_grad_norm = False

    return args


class Descriptor(nn.Module):
    def __init__(self):
        super(Descriptor, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, 15, 1, 4, bias=True)
        self.conv2 = nn.Conv2d(100, 64, 5, 1, 2, bias=True)
        self.conv3 = nn.Conv2d(64, 30, 3, 1, 2, bias=True)

    def forward(self, x):
        # TODO adjust the model structure
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), (3,3))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))

        return x.squeeze()


class Model(nn.Module):
    def __init__(self, logger, opts, device):
        super(Model, self).__init__()
        self.logger = logger
        self.opts = opts
        self.device = device

    def langevin(self, descriptor, x):
        eps = self.opts.langevin_step_size
        s = self.opts.sigma
        for i in range(self.opts.langevin_step_num):
            x = Variable(x.data, requires_grad=True)
            x_feature = descriptor(x)
            x_feature.backward(torch.ones_like(x_feature)) # a tensor filled with 1 with the same size as x_feature
            noise = torch.randn_like(x).to(self.device)
            x.data += eps * eps / 2 * x.grad - eps * eps / 2 / s / s * x + eps * noise
        return x

    def train(self):
        dataset = IgnoreLabelDataset(datasets.ImageFolder(root=self.opts.data_path,
                                       transform=transforms.Compose([
                                           transforms.Resize(self.opts.img_size),
                                           transforms.ToTensor()
                                       ])))

        descriptor = Descriptor().to(self.device)
        im = dataset[0].unsqueeze(0).to(self.device)
        im_mean = create_mean_image(im).to(self.device)
        num_filters = get_num_filters(descriptor, im.shape, self.device)

        sample_pos = normalize_image(im, im_mean)
        sample_neg = torch.zeros_like(sample_pos)

        save_images(sample_pos + im_mean, '{}/data.png'.format(self.opts.output_dir))

        for epoch in range(self.opts.num_epoch):

            sample_neg = self.langevin(descriptor, sample_neg)

            sample_pos_feature = descriptor(sample_pos)
            sample_neg_feature = descriptor(sample_neg)

            en_pos = sample_pos_feature.sum()
            en_neg = sample_neg_feature.sum()
            loss = en_pos - en_neg

            descriptor.zero_grad()
            loss.backward()
            grad_norm = 0.
            for p, n_f, lr in zip(descriptor.parameters(), num_filters, self.opts.lr):
                params_norm = torch.norm(p.data)
                grad = p.grad.data / n_f
                grad_norm_per_layer = torch.norm(grad)

                p.data += grad * lr
                if (self.opts.debug_grad_norm):
                    self.logger.info('layer param norm = {:>18.4f} original grad norm = {:>18.4f}'.format(params_norm, grad_norm_per_layer))

                grad_norm += grad_norm_per_layer

            self.logger.info('{:>5d} loss={:>18.2f} en(pos)={:>18.2f} en(neg)={:>18.2f} norm(grad)={:>18.6f}'.format(epoch, loss, en_pos.sum(), en_neg.sum(), grad_norm))

            if epoch % 10 == 0:
                # self.logger.info('max= {:>10.2f}, min= {:>10.2f}'.format(torch.max(sample_neg + im_mean), torch.min(sample_neg + im_mean)))
                save_images(rescaleSynthesizedImage(sample_neg + im_mean), '{}/{}.png'.format(self.opts.output_dir, epoch))

            if epoch > 0 and epoch % 1000 == 0:
                torch.save(descriptor.state_dict(), self.opts.ckpt_dir + '/descriptor_{}.pth'.format(epoch))

        saveConv1Filters(self.opts.ckpt_dir, descriptor)


def rescaleSynthesizedImage(img):
    img[img<0] = 0
    img[img>255] = 255
    return img


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


def create_mean_image(im):
    im_mean = torch.zeros(im.shape)
    means = im.mean(-1).mean(-1)[0]
    im_mean[:, 0, :, :] = means[0] * 255
    im_mean[:, 1, :, :] = means[1] * 255
    im_mean[:, 2, :, :] = means[2] * 255
    # im_mean[:, 0, :, :] = 123.680
    # im_mean[:, 1, :, :] = 116.779
    # im_mean[:, 2, :, :] = 103.939
    return im_mean


def get_num_filters(net, shape, device):
    x = torch.zeros(shape).to(device)
    num_filters = torch.zeros(len(list(net.parameters())), dtype=torch.float).to(device)
    for param in net.parameters():
        print(type(param.data), param.size())
    for c in net.children():
        print(c)
    for i, c in enumerate(net.children()):
        x = c(x)
        num_filters[i*2+0] = x.shape[2] * x.shape[3]
        num_filters[i*2+1] = x.shape[2] * x.shape[3]
        # num_filters[i] = x.shape[2] * x.shape[3]
    return num_filters


def normalize_image(im, mean_im):
    im.data *= 255
    im.data -= mean_im.data
    return im


def grad_norm(net):
    return torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def set_cudnn():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def set_gpu(device):
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


def save_images(img, path):
    vutils.save_image(img, path, normalize=True, nrow=1)


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/{}'.format(exp_id), t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def main():
    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = get_output_dir(exp_id)
    opt = opts(output_dir)

    set_seed(1)
    set_cudnn()
    set_gpu(opt.gpu)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    logger = setup_logging('main', opt.log_dir)
    copy_source(__file__, opt.ckpt_dir)

    model = Model(logger, opt, device)
    model.train()


def loadModel(dir, filename):
    descriptor = Descriptor()
    descriptor.load_state_dict(torch.load(os.path.join(dir, filename)))
    return descriptor

def saveConv1Filters(dir, descriptor, filename='layer1_filters.png'):
    weights = list(descriptor.parameters())[0].data
    vutils.save_image(weights[0:64, :, :, :], os.path.join(dir, filename), normalize=True)


if __name__ == '__main__':
    main()
