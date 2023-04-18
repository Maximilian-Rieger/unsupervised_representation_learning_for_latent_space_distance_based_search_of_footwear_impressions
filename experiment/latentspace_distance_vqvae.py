import argparse
import torch
import torch.nn.init
import torch.backends.cudnn as cudnn
import os
import logging
from utils.utils import GPU
from torchvision import transforms as Transforms
from utils.utils import load_config
from utils.utils import dynamic_import

from latentspace_distance import load_image


def calculate_distance(img_a, img_b, encoder, transforms, dist=torch.dist):
    image_a = load_image(img_a, transforms)
    image_b = load_image(img_b, transforms)

    image_a = image_a.to(GPU.device).unsqueeze_(1)
    encoding_a = encoder(image_a).view(-1).detach()
    image_b = image_b.to(GPU.device).unsqueeze_(1)
    encoding_b = encoder(image_b).view(-1).detach()

    return dist(encoding_a, encoding_b).item()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')

    # Training settings
    parser = argparse.ArgumentParser(description='Impress latent space distance')
    parser.add_argument('images', type=str, metavar='IMG', nargs=2,
                        help='jsonPath to image')
    parser.add_argument('--model', default=None, required=True, type=str, metavar='MODEL',
                        help='name of model to use')
    parser.add_argument('--model-weights', default=None, required=True, type=str, metavar='WEIGHT_PATH',
                        help='jsonPath to checkpoint of model')
    parser.add_argument('--config', required=True, type=str, metavar='CONFIG', nargs='+',
                        help='path to config file with model config')
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
    if 'model_weights' not in args:
        raise AssertionError('model-weights to use must be passed')

    model = args['model']
    modelWeights = args['model_weights']

    configArgs = load_config(args) if 'config' in args else {'model': {}}

    Experiment = dynamic_import(model, 'Experiment')

    vqvae, = Experiment.load_model(modelWeights, **configArgs['model'])
    vqvae.eval()
    transforms = Transforms.Compose([
        Transforms.Grayscale(),
        Transforms.ToTensor(),
    ])

    img_a, img_b = args['images']
    distance = calculate_distance(img_a, img_b, lambda x: vqvae.encode(x)[0], transforms)

    logging.info('Latentspace distance is {}'.format(distance))




if __name__ == '__main__':
    main()
