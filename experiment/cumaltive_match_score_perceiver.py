import logging
import os, glob
import argparse

import torch
import torch.nn.init
import torch.backends.cudnn as cudnn
from utils.utils import GPU
from torchvision import transforms as Transforms
from tqdm import tqdm
import matplotlib as mpl
from utils.utils import load_config
from latentspace_distance_perceiver import calculate_distance
from cumaltive_match_score import print_matrix, cms_full, plot_cms
from utils.utils import dynamic_import
from einops import rearrange

mpl.use('Agg')


# mit cumulative match score evalurieren
def match_matrix(data, encoder, transforms, dist=torch.dist):
    c = len(data)
    pbr = tqdm(data)
    pbr.set_description('Match Matrix')
    res = [[0 for x in range(c)] for y in range(c)]
    res_ = [[0 for x in range(c)] for y in range(c)]
    for i, img_a in enumerate(pbr):
        for n, img_b in enumerate(data):
            res[i][n] = (calculate_distance(img_a, img_b, encoder, transforms, dist=dist), n, i == n)
    for i, line in enumerate(res):
        res_[i] = sorted(line, key=lambda item: item[0])
    return res_


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')

    # Training settings
    parser = argparse.ArgumentParser(description='Impress Cumulative Match Score')
    parser.add_argument('--dataset', type=str, metavar='DATASET', help='jsonPath to datasetRoot')
    parser.add_argument('--model', default=None, required=True, type=str, metavar='MODEL',
                        help='name of model to use')
    parser.add_argument('--model-weights', default=None, required=True, type=str, metavar='WEIGHT_PATH',
                        help='jsonPath to checkpoint of model')
    parser.add_argument('--config', required=True, type=str, metavar='CONFIG', nargs='+',
                        help='path to config file with model config')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--gpuid', default='-1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args().__dict__

    args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()

    if args['cuda']:
        GPU.set(args['gpuid'], 400)
        cudnn.benchmark = True

    if 'model' not in args:
        raise AssertionError('a model to use must be passed')

    model = args['model']
    modelWeights = args['model_weights']
    configArgs = load_config(args) if 'config' in args else {'model': {}}

    data_r = os.path.join(args['dataset'], '*', '*_1_R.jpg')
    data_r = glob.glob(data_r)
    data_l = os.path.join(args['dataset'], '*', '*_3_L.jpg')
    data_l = glob.glob(data_l)
    data = data_r + data_l
    data_len = len(data)
    logging.info('Calculating cms for {} images'.format(data_len))

    Experiment = dynamic_import(model, 'Experiment')
    perceiver, _, _ = Experiment.load_model(modelWeights, configArgs['batchsize'], **configArgs['model'])
    perceiver.eval()
    transforms = Transforms.Compose([
        Transforms.Grayscale(),
        Transforms.Resize((256, 256)),
        Transforms.ToTensor(),
        Transforms.Lambda(lambd=lambda x: rearrange(x, 'c h w -> h w c'))
    ])

    matrix = match_matrix(data, perceiver, transforms)

    try:
        print_matrix(matrix, 400)
    except AttributeError as e:
        logging.info('error while printing matrix')
        logging.error(e)

    res = cms_full(matrix)
    # res = [36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # res = make_steps(res)

    plot_cms(res, 'cms.fig.perceiver.png')
    print()
    print(res)


if __name__ == '__main__':
    main()
