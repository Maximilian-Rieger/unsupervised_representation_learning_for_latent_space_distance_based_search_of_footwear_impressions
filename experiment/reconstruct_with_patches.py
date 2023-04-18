import argparse
import torch
import torch.nn.init
import torch.backends.cudnn as cudnn
import os
import logging
from utils.utils import GPU
from torchvision import transforms as Transforms
from PIL import Image  # reading image
from experiment.experiment_aae import Experiment
from dataloading.transforms import SlidingWindowTransformClass, CombiningWindowTransformClass


def load_image(image, transforms):
    image = Image.open(image)
    width, height = image.size
    image = transforms(image)
    return image, width, height


def save_image(image, path):
    image.save(path, 'JPEG')


def encode_image(img, encoder, transforms, batch_size=None):
    patched_image, width, height = load_image(img, transforms)
    patch_count = len(patched_image)

    encoding = []
    for p in range(patch_count):
        patch = patched_image[p, :, :]
        patch = patch.to(GPU.device).unsqueeze_(0).unsqueeze_(0)
        patch = encoder(patch)
        patch = patch.detach()
        encoding += [patch]
    return encoding, width, height


def decode_image(encoding, decoder, transforms, latent_size, batch_size=None):
    img = []
    for patch in encoding:
        patch = patch.view(1, latent_size)
        patch = patch.to(GPU.device)
        patch = decoder(patch).detach()
        img += [patch]
    decoded = transforms(img)
    return decoded


def encode_image_batched(img, encoder, transforms, batch_size=19):
    logging.info('\tPatching started...')
    patched_image, width, height = load_image(img, transforms)
    logging.info('\tPatching done...')
    patch_count = len(patched_image)

    encoding = []

    logging.info('\tPatch encoding started...')
    for p in range(0, patch_count, batch_size):
        img = patched_image[p:p + batch_size, :, :]
        img = img.to(GPU.device).unsqueeze_(1)
        img = encoder(img).view(batch_size, -1).detach()
        encoding += [img]
    logging.info('\tPatch encoding done...')

    return encoding, width, height


def decode_image_batched_old(encoding, decoder, transforms, latent_size, batch_size=19):
    img = []
    patch_count = len(encoding)
    logging.info('Patch decoding started...')
    for p in range(0, patch_count, batch_size):
        patch = encoding[p:p + batch_size]
        patch = patch.view(1, latent_size)
        patch = patch.to(GPU.device)
        patch = decoder(patch).detach()
        img += [patch]
    logging.info('Patch decoding done...')

    logging.info('DePatching started...')
    decoded = transforms(img)
    logging.info('DePatching done...')

    return decoded


def decode_image_batched(encoding, decoder, transforms, latent_size, batch_size=19):
    img = []
    logging.info('\tPatch decoding started...')
    for patch in encoding:
        patch = patch.view(batch_size, latent_size)
        patch = patch.to(GPU.device)
        patch = decoder(patch).detach()
        for n in range(batch_size):
            patch_single = patch[n, 0]
            img += [patch_single]
    logging.info('\tPatch decoding done...')

    logging.info('\tDePatching started...')
    decoded = transforms(img)
    logging.info('\tDePatching done...')

    return decoded


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')

    # Training settings
    parser = argparse.ArgumentParser(description='Impress reconstruction')
    parser.add_argument('image', type=str, metavar='IMG',
                        help='jsonPath to image')
    parser.add_argument('--model', default=None, required=True, type=str, metavar='PATH',
                        help='jsonPath to checkpoint of model')
    parser.add_argument('--latent_size', default=100, required=True, type=int, metavar='LATENT_SIZE',
                        help='latentspace size the model uses (default: 100)')
    parser.add_argument('--img_shape', default=256, required=True, type=int, metavar='IMG_SHAPE',
                        help='image size the model uses (default: 256)')
    parser.add_argument('--batch_size', default=19, type=int, metavar='BATCH_SIZE',
                        help='batch size the model should use (default: 19)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpuid', default='-1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--difference', default=False, type=bool,
                        help='Create extra difference img')
    args = parser.parse_args().__dict__

    args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()

    if args['cuda']:
        GPU.set(args['gpuid'], 400)
        cudnn.benchmark = True

    if 'model' not in args:
        raise AssertionError('a model to use must be passed')

    encode = encode_image_batched
    decode = decode_image_batched
    batch_size = args['batch_size']
    if batch_size == 0:
        encode = encode_image
        decode = decode_image

    encoder, decoder, discriminator = Experiment.load_model(args['model'], args['img_shape'], args['latent_size'], show_summary=False)
    encoder.eval()
    decoder.eval()
    encoding_transforms = Transforms.Compose([
        Transforms.ToTensor(),
        SlidingWindowTransformClass(args['img_shape'], args['img_shape']),
    ])

    img = args['image']

    logging.info('Encoding started...')
    encoding, width, height = encode(img, encoder, encoding_transforms, batch_size=batch_size)
    logging.info('Encoding done!!!')

    decoding_transforms = Transforms.Compose([
        CombiningWindowTransformClass((width, height), args['img_shape']),
        Transforms.ToPILImage(),
    ])

    logging.info('Decoding started...')
    decoded = decode(encoding, decoder, decoding_transforms, args['latent_size'], batch_size=batch_size)
    logging.info('decoding done!!!')

    recon_path = img + '.recon.jpg'
    save_image(decoded, recon_path)
    logging.info('Saved reconstruction to: {}'.format(recon_path))

    if args['difference']:
        logging.info('Starting difference calculation...')
        loading_transform = Transforms.ToTensor()
        image, width, height = load_image(img, loading_transform)
        decoded = loading_transform(decoded)
        image = decoded - image
        image = Transforms.ToPILImage()(image)
        logging.info('difference calculation done!!!')

        diff_path = img + '.diff.jpg'
        save_image(image, diff_path)
        logging.info('Saved difference to: {}'.format(diff_path))


if __name__ == '__main__':
    main()
