import argparse
import torch
import torch.nn.init
import torch.backends.cudnn as cudnn
import os
import logging
from utils.utils import GPU
from torchvision import transforms as Transforms
from PIL import Image  # reading image
# from experiment.experiment_aae import Experiment
from dataloading.transforms import SlidingWindowTransformClass
from utils.utils import dynamic_import


def load_image(image, transforms=None):
    image = Image.open(image)
    if transforms:
        image = transforms(image)
    return image


def torchNeumaierSum(sequence):
    acc = torch.zeros(1).to(GPU.device)
    c = torch.zeros(1).to(GPU.device)  # A running compensation for lost low-order bits.
    for i in range(len(sequence)):
        t = acc + sequence[i]
        if acc.abs() >= sequence[i].abs():
            c += (acc - t) + sequence[i]  # If sum is bigger, low-order digits of input[i] are lost.
        else:
            c += (sequence[i] - t) + acc  # Else low-order digits of sum are lost.
        acc = t
    return acc + c


def calculate_distance_neumaier(img_a, img_b, encoder, transforms):
    patched_image_a = load_image(img_a, transforms)
    patched_image_b = load_image(img_b, transforms)
    patch_count_a = len(patched_image_a)
    patch_count_b = len(patched_image_b)
    assert patch_count_a == patch_count_b, 'images have to be patched into same amount of patches'

    encoding_sum = torch.zeros(1, requires_grad=False).to(GPU.device)
    encoding_correction = torch.zeros(1, requires_grad=False).to(GPU.device)
    for p in range(patch_count_a):
        img_a = patched_image_a[p, :, :]
        img_a = img_a.to(GPU.device).unsqueeze_(0).unsqueeze_(0)
        img_a = encoder(img_a).view(-1).detach()
        img_b = patched_image_b[p, :, :]
        img_b = img_b.to(GPU.device).unsqueeze_(0).unsqueeze_(0)
        img_b = encoder(img_b).view(-1).detach()
        img_a -= img_b
        img_a.pow_(2)
        img_a = torchNeumaierSum(img_a)

        img_b = encoding_sum + img_a
        if encoding_sum.abs() >= img_a.abs():
            encoding_correction += (encoding_sum - img_b) + img_a  # If sum is bigger, low-order digits of input[i] are lost.
        else:
            encoding_correction += (img_a - img_b) + encoding_sum  # Else low-order digits of sum are lost.
        encoding_sum = img_b

    del patched_image_a, patched_image_b
    return encoding_sum.sqrt_().cpu().item()


def calculate_patched_distance_(img_a, img_b, encoder, transforms):
    patched_image_a = load_image(img_a, transforms)
    patched_image_b = load_image(img_b, transforms)
    patch_count_a = len(patched_image_a)
    patch_count_b = len(patched_image_b)
    assert patch_count_a == patch_count_b, 'images have to be patched into same amount of patches'

    encoding_sum = torch.zeros(1, requires_grad=False).to(GPU.device)
    for p in range(patch_count_a):
        img_a = patched_image_a[p, :, :]
        img_a = img_a.to(GPU.device).unsqueeze_(0).unsqueeze_(0)
        img_a = encoder(img_a).view(-1).detach()
        img_b = patched_image_b[p, :, :]
        img_b = img_b.to(GPU.device).unsqueeze_(0).unsqueeze_(0)
        img_b = encoder(img_b).view(-1).detach()
        img_a -= img_b
        img_a.pow_(2)
        img_a = img_a.sum()
        encoding_sum += img_a

    del patched_image_a, patched_image_b
    return encoding_sum.sqrt_().cpu().item()


def calculate_patched_distance(img_a, img_b, encoder, transforms):
    patched_image_a = load_image(img_a, transforms)
    patched_image_b = load_image(img_b, transforms)
    patch_count_a = len(patched_image_a)
    patch_count_b = len(patched_image_b)
    assert patch_count_a == patch_count_b, 'images have to be patched into same amount of patches'

    encoding_a = []
    encoding_b = []
    for p in range(patch_count_a):
        img_a = patched_image_a[p, :, :]
        img_a = img_a.to(GPU.device).unsqueeze_(0).unsqueeze_(0)
        img_a = encoder(img_a).view(-1).detach()
        encoding_a += [img_a]
        img_b = patched_image_b[p, :, :]
        img_b = img_b.to(GPU.device).unsqueeze_(0).unsqueeze_(0)
        img_b = encoder(img_b).view(-1).detach()
        encoding_b += [img_b]
    encoding_a = torch.stack(encoding_a)
    encoding_b = torch.stack(encoding_b)
    return torch.dist(encoding_a, encoding_b).item()


def calculate_patched_distance_batched(img_a, img_b, encoder, transforms, batch_size=19, dist=torch.dist):
    patched_image_a = load_image(img_a, transforms)
    patched_image_b = load_image(img_b, transforms)
    patch_count_a = len(patched_image_a)
    patch_count_b = len(patched_image_b)
    assert patch_count_a == patch_count_b, 'images have to be patched into same amount of patches'

    encoding_a = []
    encoding_b = []
    for p in range(0, patch_count_a, batch_size):
        img_a = patched_image_a[p:p+batch_size, :, :]
        img_a = img_a.to(GPU.device).unsqueeze_(1)
        img_a = encoder(img_a).view(-1).detach()
        encoding_a += [img_a]
        img_b = patched_image_b[p:p+batch_size, :, :]
        img_b = img_b.to(GPU.device).unsqueeze_(1)
        img_b = encoder(img_b).view(-1).detach()
        encoding_b += [img_b]
    encoding_a = torch.stack(encoding_a)
    encoding_b = torch.stack(encoding_b)
    return dist(encoding_a, encoding_b).item()


def calculate_distance(img_a, img_b, encoder, transforms):
    image_a = load_image(img_a, transforms)
    image_b = load_image(img_b, transforms)

    image_a = image_a.to(GPU.device).unsqueeze_(1)
    encoding_a = encoder(image_a).view(-1).detach()
    image_b = image_b.to(GPU.device).unsqueeze_(1)
    encoding_b = encoder(image_b).view(-1).detach()

    return torch.dist(encoding_a, encoding_b).item()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')

    # Training settings
    parser = argparse.ArgumentParser(description='Impress latent space distance')
    parser.add_argument('images', type=str, metavar='IMG', nargs=2,
                        help='jsonPath to image')
    parser.add_argument('--model', default=None, required=True, type=str, metavar='model',
                        help='name of model to use')
    parser.add_argument('--model-weights', default=None, required=True, type=str, metavar='WEIGHT_PATH',
                        help='jsonPath to checkpoint of model')
    parser.add_argument('--latent_size', default=100, required=True, type=int, metavar='LATENT_SIZE',
                        help='latentspace size the model uses (default: 100)')
    parser.add_argument('--img_shape', default=256, required=True, type=int, metavar='IMG_SHAPE',
                        help='image size the model uses (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpuid', default='-1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args().__dict__

    args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()

    if args['cuda']:
        GPU.set(args['gpuid'], 400)
        cudnn.benchmark = True

    if 'model' not in args:
        raise AssertionError('a model to use must be passed')
    if 'model-weights' not in args:
        raise AssertionError('model-weights to use must be passed')

    model = args['model']
    modelWeights = args['model-weights']

    Experiment = dynamic_import(model, 'Experiment')

    encoder, decoder, discriminator = Experiment.load_model(modelWeights, args['img_shape'], args['latent_size'], show_summary=False)
    encoder.eval()
    transforms = Transforms.Compose([
        Transforms.ToTensor(),
        SlidingWindowTransformClass(args['img_shape'], args['img_shape']),
    ])

    img_a, img_b = args['images']
    distance = calculate_patched_distance_batched(img_a, img_b, encoder, transforms, batch_size=19)

    logging.info('Latentspace distance is {}'.format(distance))

# 1_3_L / 1_4_L 1166.531494140625
# 1_3_L / 1_c_L 1459.9449462890625
# 1_3_L / 49_3_L 1320.102783203125
# 1_3_L / 70_3_L 1380.3966064453125


if __name__ == '__main__':
    main()
