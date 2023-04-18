import logging
import os, glob
import argparse

import torch
import torch.nn.init
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from experiment.data_zoo import DataZoo
from torch.utils.data import DataLoader
from utils.utils import GPU
from torchvision import transforms as Transforms
from tqdm import tqdm
import matplotlib as mpl
from utils.utils import load_config
from cumaltive_match_score import print_matrix, cms_full, plot_cms
from utils.utils import dynamic_import
from einops import rearrange
import json

# mpl.use('Agg')
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def evaluate_match_matrix(matrix):
    c = len(matrix)
    pbr = tqdm(matrix)
    pbr.set_description('Evaluating match matrix')
    res = {}
    for i, line in enumerate(pbr):
        if line[0][4] == line[0][5]:
            res.update({str(line[0][4].item()): {
                'dist': line[0][0].item(),
                'seq_dist': 0,
                'i': line[0][1],
                'n': line[0][2],
                'direct_match': str(line[0][4].item()),
                'label_match': (line[0][4].item() == line[0][5].item()),
                'label_a': str(line[0][4].item()),
                'label_b': str(line[0][5].item())
            }})
        else:
            found = False
            for n, item in enumerate(line):
                if item[4] == item[5]:
                    res.update({str(item[4].item()): {
                        'dist': item[0].item(),
                        'seq_dist': n,
                        'i': item[1],
                        'n': item[2],
                        'direct_match': str(line[0][5].item()),
                        'label_match': (item[4].item() == item[5].item()),
                        'label_a': str(item[4].item()),
                        'label_b': str(item[5].item())
                    }})
                    found = True
                    break
            if not found:
                res.update({str(line[0][4].item()): {
                    'dist': float('inf'),
                    'seq_dist': c,
                    'i': line[0][1],
                    'n': line[0][2],
                    'direct_match': str(line[0][5].item()),
                    'label_match': False,
                    'label_a': line[0][4].item(),
                    'label_b': line[0][5].item()
                }})

    return res


def make_data_from_evl(evl):
    count = len(evl) * 2
    x = torch.arange(count)
    y = torch.zeros(count, dtype=torch.int)
    for key, value in evl.items():
        y[value['seq_dist']] = y[value['seq_dist']] + 1
    # make cumulative
    y = y.cumsum(0)
    y = y / (count * 0.5)
    return x, y


def match_matrix(data, encoder, dist=torch.dist, filter_exact_match=False):
    c = len(data)
    pbr = tqdm(data)
    pbr.set_description('Encoding images')
    encodings = [encode_image(batch, encoder) for batch in pbr]
    pbr = tqdm(encodings)
    pbr.set_description('Calculating distances')
    res = [[0 for x in range(c)] for y in range(c)]
    res_ = [[0 for x in range(c)] for y in range(c)]
    if not filter_exact_match:
        for i, encoding_a in enumerate(pbr):
            for n, encoding_b in enumerate(encodings):
                res[i][n] = (dist(encoding_a[1], encoding_b[1]), i, n, i == n, encoding_a[0], encoding_b[0])
    else:
        for i, encoding_a in enumerate(pbr):
            for n, encoding_b in enumerate(encodings):
                if i != n:
                    res[i][n] = (dist(encoding_a[1], encoding_b[1]), i, n, i == n, encoding_a[0], encoding_b[0])
                else:
                    res[i][n] = (float('inf'), i, n, i == n, encoding_a[0], encoding_b[0])
    pbr = tqdm(res)
    pbr.set_description('Sorting distances')
    for i, line in enumerate(pbr):
        res_[i] = sorted(line, key=lambda item: item[0])
    return res_


def encode_image(data, encoder):
    label, image = data
    image = image.to(GPU.device)
    return (label, encoder(image).view(-1).detach())


def un_kwp(**kwargs):
    return [value for (key, value) in kwargs.items()]


def sim_dist(a, b, dim=-1, eps: float = 1e-8):
    # return 1 - nn.CosineSimilarity(dim=-1)(a, b)
    return 1 - F.cosine_similarity(a, b, dim, eps)


def load_options(path):
    options_file = os.path.join(path, "options.json")
    if not os.path.exists(options_file):
        return None
    with open(options_file, "r") as f:
        options = json.load(f)
    return options


dists = {
    'cosine': sim_dist,
    'euclid': torch.dist,
}


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
    parser.add_argument('--recompute', action='store_true', default=False, help='force recompute')
    parser.add_argument('--dist', type=str, metavar='DISTANCE', default='euclid_dist', help='Distance metric')
    parser.add_argument('--gpuid', default='-1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args().__dict__

    args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()

    if args['cuda']:
        GPU.set(args['gpuid'], 400)
        cudnn.benchmark = True

    if 'model' not in args:
        raise AssertionError('a model to use must be passed')
    evl = None
    calculate = False
    try:
        with open('match_stat.json', 'r') as f:
            pass
    except FileNotFoundError:
        calculate = True
    calculate = args['recompute'] or calculate

    model = args['model']
    modelWeights = args['model_weights']
    experiment_name = modelWeights.split('\\')[-1]
    if calculate:
        configArgs = load_options(modelWeights)

        if configArgs is None:
            configArgs = load_config(args) if 'config' in args else {'model': {}}

        img_size = 128
        transforms = Transforms.Compose([
            Transforms.Grayscale(),
            Transforms.Resize((img_size, img_size)),
            Transforms.ToTensor(),
            # Transforms.Lambda(lambd=lambda x: rearrange(x, 'c h w -> h w c'))
        ])

        data_options = {
            "base": args['dataset'],
            "dataset": "Impress_2",
            "set": "clean",
            'cache': True,
            'return_path': True,
            "transform": transforms,
        }

        data = DataZoo.get(**data_options)
        data_len = len(data)
        dataLoader = DataLoader(data)
        logging.info('Calculating cms for {} images'.format(data_len))

        Experiment = dynamic_import(model, 'Experiment')
        mpl.use('TkAgg')
        img_shape = Experiment.get_img_shape(configArgs)
        # encoder, _ = Experiment.load_model(modelWeights, img_shape=img_shape, batch_size=1, **configArgs['model'])
        encoder, _ = Experiment.load_model(modelWeights, img_shape=img_shape, **configArgs['model'])
        encoder.eval()

        dist = dists[args['dist']]

        matrix = match_matrix(dataLoader, encoder, filter_exact_match=True, dist=dist)
        # matrix = match_matrix(dataLoader, encoder, filter_exact_match=True, dist=torch.dist)

        evl = evaluate_match_matrix(matrix)

        # save as json
        with open('match_stat.json', 'w') as f:
            json.dump(evl, f)
    else:
        print('match_stat.json exists\nskipping calculation')
        with open('match_stat.json', 'r') as f:
            evl = json.load(f)

    # calculate cms
    x, y = make_data_from_evl(evl)
    # plot x, y with matplotlib
    plt.plot(x, y)
    # plot 45 degree line
    line = torch.arange(0, 1.10, 0.10)
    plt.plot(line * len(x), line, '--')
    plt.title(f'Impress Cumulative Match Score\n{experiment_name}')
    plt.xlabel('% of Images retrieved')
    plt.xticks(range(0, len(x), len(x) // 10), line.numpy())

    plt.ylabel('% CMS')
    plt.yticks(line)
    # plt.xticks(torch.arange(0, 110, 10))
    plt.grid(True)

    plt.savefig('cms.png')
    plt.show()


if __name__ == '__main__':
    main()
