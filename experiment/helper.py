import os
import re
import math
import string

import matplotlib as mpl
import numpy as np
import torch
import torch.autograd as autograd
import logging
import gc

import scipy.misc
import tensorflow as tf
import enum
from PIL import Image
from matplotlib.lines import Line2D
from torchvision.transforms import transforms
import torch.nn as nn

from utils.utils import GPU

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def create_image_figure(image, title, nrows=1, ncols=1):
    fig, subplots = plt.subplots(nrows=nrows, ncols=ncols)  # type: Figure, list
    image = np.moveaxis(image.numpy(), 0, -1)
    if not (isinstance(subplots, list) or isinstance(subplots, np.ndarray)):
        subplots = [subplots]
    subplots[0].imshow(image)
    subplots[0].set_title(title)
    return fig, subplots


def create_images_figure(images, cols=1, titles=None):
    """Creates a figure of a list of data with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols: (Default = 1) Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols)).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    rows = np.ceil(n_images / float(cols))
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()  # type: Figure
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, rows, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(fig.get_size_inches()[0] * n_images * 0.6, rows)
    return fig


def normalize_img(image):
    return (image - np.min(image)) / np.ptp(image)


def normalize_img_tensor(image):
    image = image + 1  # convert to [0 ,2]
    # convert to [0 ,1]
    image = image - image.min()
    image = image / (image.max() - image.min())
    return image


# mit fixen werten probieren 0, 1
def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def add_noise(img, amplitude=0.2, cuda=True):
    noise = torch.randn(img.size())
    if cuda:
        noise = noise.cuda(GPU.device)
    noise *= amplitude
    noisy_img = img + noise
    return noisy_img


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "{} {}".format(s, size_name[i])


def save_checkpoint(epoch, train_step, models, optimizers, log_dir, limit=None, aborted=False, best=False, custom_name=None):
    checkpoint = epoch
    if limit is not None:
        checkpoint = (epoch % limit) if limit else epoch
    if aborted:
        epoch = f'{epoch}_aborted'
        checkpoint = 'aborted'
    if best:
        epoch = f'{epoch}_best'
        checkpoint = 'best'
    if custom_name is not None:
        epoch = f'{epoch}_{custom_name}'
        checkpoint = custom_name

    torch.save({
        'epoch': epoch,
        'train_step': train_step,
        'state_dict': [model.state_dict() for model in models],
        'optim_dict': [optimizer.state_dict() for optimizer in optimizers],
    }, f'{log_dir}/checkpoint_{checkpoint}.pth')


def forward_argument(dictionary, argument, sets=None):
    assert argument in dictionary, 'Forwarded argument "{}" must be in dictionary'.format(argument)
    for s in sets:
        assert s in dictionary, 'Forwarding set "{}" must be in dictionary'.format(s)
        if dictionary[argument]:
            dictionary[s][argument] = dictionary[argument]
    return dictionary


def forward_arguments(dictionary, arguments, sets=None):
    for argument in arguments:
        dictionary = forward_argument(dictionary, argument, sets)
    return dictionary


def find_checkpoints(rootdir, regex=None):
    if regex is None:
        regex = re.compile(r'checkpoint_(\d+).pth$|checkpoint_best.pth$')
    matched = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                matched.append(file)
    return matched


#
# def overlay(base, prediction, channel=0, in_place=False, resize='crop'):
#     assert resize in ('crop', 'invcrop', 'scale', 'invscale'), 'Resize has to be one of ["crop", "invcrop", "scale", "invscale"]'
#     overlay_img = base
#     if not in_place:  # copy image data
#         overlay_img = base.copy()
#
#     if resize == 'crop':
#         crop_diff = base.shape[0] - prediction.shape[0]
#         start = int(crop_diff / 2)
#         end = prediction.shape[0] + start
#         # center crop base to prediction size
#         overlay_img = overlay_img[start:end, start:end, :]
#     elif resize == 'invcrop':
#         pad_width = math.ceil((base.shape[0] - prediction.shape[0]) / 2)
#
#         # pad prediction to base size
#         prediction = np.pad(prediction, pad_width, 'constant', constant_values=0)
#     elif resize == 'scale':
#         overlay_img = resize_numpy_image(overlay_img, prediction.shape)
#     elif resize == 'invscale':
#         prediction = resize_numpy_image(prediction, base.shape)
#
#     # overlay prediction on channel
#     overlay_img[:, :, channel] = np.clip(overlay_img[:, :, channel] + prediction, 0, 1)
#     return overlay_img
#
#
# def resize_numpy_image(img, shape, resize_mode=Image.NEAREST):
#     img = img * 255
#     img = img.astype(np.uint8)
#     img = Image.fromarray(img)
#     img = img.resize(shape[0:2], resize_mode)
#     img = np.array(img)
#
#     img = img / 255
#     return img


def normalize(img):
    min = torch.min(img)
    range = torch.max(img) - min
    if range > 0:
        normalised = (img - min) / range
    else:
        normalised = torch.zeros(img.size())
    return normalised


def apply_color_map(img, cmap_name='gist_eart'):
    return plt.get_cmap(cmap_name)(img, bytes=True)


def legend(cmap_name, legend_h, legend_w):
    return plt.get_cmap(cmap_name)(np.linspace(0, 1, legend_h).repeat(legend_w).reshape(legend_h, legend_w), bytes=True)


def cuda_use():
    gpu_id = GPU.device.index
    return torch.cuda.max_memory_cached(gpu_id), torch.cuda.max_memory_allocated(gpu_id)


def print_cuda_debug():
    gpu_id = GPU.device.index
    cached, allocated = [convert_size(memory) for memory in cuda_use()]
    logging.info('GPU: {} Max Memory cached: {} Max Memory allocated: {}'.format(gpu_id, cached, allocated))


def memory_report():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            logging.info('Tensor: {}, size: {}'.format(type(obj), obj.size()))


class CallableGenerator:
    def __init__(self, generator):
        self.generator = generator

    def __call__(self):
        return self.generator.next()


def save_images_from_event(fn: string, output_dir: string = './', name_regex: string = r'(\w+-\d{4}(?:-\d\d)+)') -> None:
    assert (os.path.isdir(output_dir))
    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    prefix = re.search(name_regex, fn).group()
    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                try:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                except tf.errors.InvalidArgumentError:
                    logging.debug('Error while parsing tag {}'.format(v.tag))
                    continue
                output_fn = os.path.realpath('{}/{}_image_{:05d}.png'.format(output_dir, prefix, count))
                logging.info("Saving '{}'".format(output_fn))
                scipy.misc.imsave(output_fn, im)
                count += 1


def compute_out_size(in_size, module):
    """
    Compute output size of Module `module` given an x with size `in_size`.
    Example:
        >>>net = torchvision.models.googlenet()
        >>>x = torch.Tensor(1, 3, 224, 224)  # shape = (batch size, channels, height, width)
        >>>print(compute_size(x.size(), net.forward))

    """
    f = module.forward(autograd.Variable(torch.Tensor(1, *in_size)), ),
    return int(np.prod(f.size()[1:]))


def load_checkpoint(models, optimizers, args):
    train_step = 0
    path = args['resume']
    if 'log_dir' in args:
        path = args['log_dir'] + os.sep + path

    if os.path.isfile(path):
        logging.debug('=> loading checkpoint "{}"'.format(path))
        checkpoint = torch.load(path, GPU.device)

        for model, state_dict in zip(models, checkpoint['state_dict']):
            model.load_state_dict(state_dict)
        if optimizers is not None and 'optim_dict' in checkpoint:
            for optimizer, optim_dict in zip(optimizers, checkpoint['optim_dict']):
                optimizer.load_state_dict(optim_dict)

        args['start_epoch'] = checkpoint['epoch']
        train_step = checkpoint['train_step']
        logging.debug('=> loaded checkpoint "{}" (epoch {})'.format(args['resume'], checkpoint['epoch']))
        return models, optimizers, args, train_step
    elif os.path.isdir(path):
        checkpoints = find_checkpoints(path)
        checkpoints.sort(reverse=True)
        args['resume'] += '/{}'.format(checkpoints[0])  # use latest checkpoint
        return load_checkpoint(models, optimizers, args)
    else:
        logging.error('Provided resume argument is neither file nor directory! {}'.format(path))


class MaskToTensor(object):
    """Convert ndarray to Tensor."""

    def __call__(self, mask):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        mask = np.array(mask).astype(np.float32)
        mask = np.moveaxis(mask, -1, 0)

        mask = torch.from_numpy(mask).float()

        return mask


def plot_grad_flow_bars(named_parameters, file, alpha=0.5):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters(), filename)" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    max_max = np.max(max_grads)
    max_max = 0.1 if np.isnan(max_max) or np.isinf(max_max) else max_max
    plt.clf()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=alpha, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=alpha, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.ylim(bottom=0, top=max_max)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([
        Line2D([0], [0], color="c", lw=4),
        Line2D([0], [0], color="b", lw=4),
        Line2D([0], [0], color="k", lw=4)],
        ['max-gradient', 'mean-gradient', 'zero-gradient'],
        loc=2
    )
    plt.savefig(file, bbox_inches='tight', pad_inches=1)


def plot_grad_flow_lines(named_parameters, file, alpha=0.5):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())
    if len(max_grads) == 0:
        max_grads = [0]
    max_max = np.max(max_grads)
    max_max = 0.05 if np.isnan(max_max) or np.isinf(max_max) else max_max
    plt.clf()
    plt.plot(max_grads, alpha=alpha, color="c")
    plt.plot(ave_grads, alpha=alpha, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0, top=max_max)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([
        Line2D([0], [0], color="c", lw=4),
        Line2D([0], [0], color="b", lw=4),
        Line2D([0], [0], color="k", lw=4)],
        ['max-gradient', 'mean-gradient', 'zero-gradient'],
        loc=2
    )
    plt.savefig(file, bbox_inches='tight', pad_inches=1)


def copy_grad_info(model):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())
    return {'ave_grads': ave_grads, 'max_grads': max_grads, 'layers': layers}


def plot_multiple_grad_flow_lines(named_parameter_dict, file, alpha=0.5):
    subplots = named_parameter_dict.items()
    subplot_count = len(subplots)
    fig, plts = plt.subplots(1, subplot_count)
    if subplot_count == 1:
        plts = [plts]
    for (name, named_parameters), subplt in zip(subplots, plts):
        ave_grads = []
        max_grads = []
        layers = []
        if not isinstance(named_parameters, dict):
            for n, p in named_parameters:
                if p.requires_grad and p.grad is not None and ("bias" not in n):
                    layers.append(n)
                    ave_grads.append(p.grad.cpu().abs().mean())
                    max_grads.append(p.grad.cpu().abs().max())
        else:
            ave_grads, max_grads, layers = named_parameters['ave_grads'], named_parameters['max_grads'], named_parameters['layers']
        max_max = np.max(max_grads)
        max_max = 0.05 if np.isnan(max_max) or np.isinf(max_max) else max_max
        subplt.plot(max_grads, alpha=alpha, color="c")
        subplt.plot(ave_grads, alpha=alpha, color="b")
        subplt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        subplt.set(
            title=name,
            xlabel='Layers', ylabel='average gradient',
            xticks=range(0, len(ave_grads), 1),
            # xticklabels=layers,
        )
        subplt.set_xticklabels(layers, rotation=90)
        subplt.set_xlim(left=0, right=len(ave_grads))
        subplt.set_ylim(bottom=0, top=max_max)
        subplt.grid(True)

    fig.suptitle("Gradient flow")
    fig.legend([
        Line2D([0], [0], color="c", lw=4),
        Line2D([0], [0], color="b", lw=4),
        Line2D([0], [0], color="k", lw=4)],
        ['max-gradient', 'mean-gradient', 'zero-gradient'],
        loc=2
    )
    fig.set_size_inches(5 * subplot_count, 4)
    fig.savefig(file, bbox_inches='tight', pad_inches=1)
    plt.close(fig)


def mean_grad(named_parameters) -> np.ndarray:
    mean_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            mean_grads.append(p.grad.abs().mean())
    return np.mean(mean_grads)


def save_image(image, file):
    unloader = transforms.ToPILImage()
    image = image.cpu().clone()  # we clone the tensor to not do changes on it
    for i in range(image.shape[0]):
        img = image[i]
        if len(img.shape) == 2:
            img = img.unsqueeze(0)  # remove the fake batch dimension
        img = unloader(img)
        img.save('{}[{}].png'.format(file, i), format='png')


#
# def tensor_plot_grad():
#     with SummaryWriter(log_dir=log_dir, comment="GradTest", flush_secs=30) as writer:
#         # ... your learning loop
#         _limits = np.array([float(i) for i in range(len(gradmean))])
#         _num = len(gradmean)
#         writer.add_histogram_raw(tag=netname + "/abs_mean", min=0.0, max=0.3, num=_num,
#                                  sum=gradmean.sum(), sum_squares=np.power(gradmean, 2).sum(), bucket_limits=_limits,
#                                  bucket_counts=gradmean, global_step=global_step)
#         # where gradmean is np.abs(p.grad.clone().detach().cpu().numpy()).mean()
#         # _limits is the x axis, the layers
#         # and
#         _mean = {}
#         for i, name in enumerate(layers):
#             _mean[name] = gradmean[i]
#         writer.add_scalars(netname + "/abs_mean", _mean, global_step=global_step)

previous = None


def weights_init(m, epsilon=0.01):
    """
    Takes in a module and initializes it in the following way:
    all linear layers with xavier initialization
    all linear layers with zero bias
    all conv layers with xavier initialization
    all conv layers with zero bias if next layer is batchnorm
    all batchnorm layers with 1s
    all embedding layers with normal initialization

    :param m: module to initialize
    :param epsilon: epsilon for normal initialization
    """
    global previous
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, mean=0, std=epsilon)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(epsilon)
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(epsilon)
    elif type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(epsilon)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        if type(previous) == nn.Linear or type(previous) == nn.Conv2d or type(previous) == nn.ConvTranspose2d:
            previous.bias = None

    previous = m
