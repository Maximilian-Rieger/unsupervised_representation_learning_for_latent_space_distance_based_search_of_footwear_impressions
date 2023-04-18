import argparse
import torch
import torch.nn.init
import torch.backends.cudnn as cudnn
import os
import logging
import os, pathlib, glob
import numpy as np
from utils.utils import GPU
from torchvision import transforms as Transforms
from experiment.experiment_aae import Experiment
from experiment.latentspace_distance import calculate_patched_distance_batched, calculate_distance
from dataloading.transforms import SlidingWindowTransformClass
from tqdm import tqdm


def match_matrix_precalc():
    return torch.Tensor([[1386.6400, 1365.7637, 1371.0817, 1356.0216, 1381.1167, 1429.2458,
                          1356.2716, 1392.4023, 1447.7468, 1399.1301, 1401.5731, 1460.4998,
                          1343.5007, 1362.4177, 1422.6628, 1370.8768, 1389.0549, 1479.1403,
                          1388.7157, 1364.3348, 1333.0731, 1365.7285, 1351.2310, 1384.7158,
                          1312.5593, 1330.3734, 1331.7719, 1378.3077, 1353.1587, 1385.5516,
                          1319.1997, 0.0000, 1347.1211, 1361.0642, 1370.5781, 1437.6914]])


def match_matrix_precalc2():
    return torch.Tensor([[8.7939e+06, 4.3043e+06, 5.4266e+06, 9.5001e+06, 1.1876e+07, 1.0396e+06,
                          4.9826e+06, 8.4837e+06, 1.7622e+07, 3.8641e+08, 2.8600e+06, 1.9869e+06,
                          8.4554e+06, 1.2575e+07, 6.1058e+06, 2.4526e+06, 1.1860e+08, 4.9024e+06,
                          1.6420e+06, 3.3572e+06, 6.2803e+06, 3.7293e+06, 7.0884e+08, 6.3475e+06,
                          1.6976e+06, 7.5144e+06, 9.5723e+06, 1.0882e+06, 6.1440e+06, 3.1527e+06,
                          1.2509e+06, 0.0000e+00, 1.3154e+07, 3.1126e+06, 1.4428e+07, 2.8437e+06]])


def match_matrix(target, data, encoder, transforms):
    c = len(data)
    pbr = tqdm(data)
    pbr.set_description('Match Matrix')
    res = torch.Tensor(1, c).fill_(0)
    for i, img_b in enumerate(pbr):
        if target != img_b:
            res[0, i] = calculate_patched_distance_batched(target, img_b, encoder, transforms)
    return res


def top_k(d, n):
    return [(k, v) for k, v in sorted(d.items(), key=lambda item: item[1])[:n]]


def lsd_search(distance, data, entry, n):
    ret = {}
    target = data[entry]
    for i, image in enumerate(data):
        dist = distance(target, image)
        ret.update({i, dist})
    return top_k(ret, n)


def cmc(distance, data):
    ret = torch.zeros(len(data))
    for i in range(len(data)):
        for n in reversed(range(1, len(data))):
            topN = lsd_search(distance, data, i, n)
            if i in [k for k, v in topN]:
                ret[i, n] = 1
    return ret


# def cmc(querys, gallery, topk):
#     ret = np.zeros(topk)
#     valid_queries = 0
#     all_rank = []
#     sum_rank = np.zeros(topk)
#     for query in querys:
#         q_id = query[0]
#         q_feature = query[1]
#         # Calculate the distances for each query
#         distmat = []
#         for img, feature in gallery:
#             # Get the label from the image
#             name,_,_ = get_info(img)
#             dist = np.linalg.norm(q_feature - feature)
#             distmat.append([name, dist, img])
#
#         # Sort the results for each query
#         distmat.sort(key=custom_sort)
#         # Find matches
#         matches = np.zeros(len(distmat))
#         # Zero if no match 1 if match
#         for i in range(0, len(distmat)):
#             if distmat[i][0] == q_id:
#                 # Match found
#                 matches[i] = 1
#         rank = np.zeros(topk)
#         for i in range(0, topk):
#             if matches[i] == 1:
#                 rank[i] = 1
#                 # If 1 is found then break as you dont need to look further path k
#                 break
#         all_rank.append(rank)
#         valid_queries +=1
#     #print(all_rank)
#     sum_all_ranks = np.zeros(len(all_rank[0]))
#     for i in range(0,len(all_rank)):
#         my_array = all_rank[i]
#         for g in range(0, len(my_array)):
#             sum_all_ranks[g] = sum_all_ranks[g] + my_array[g]
#     sum_all_ranks = np.array(sum_all_ranks)
#     print("NPSAR", sum_all_ranks)
#     cmc_restuls = np.cumsum(sum_all_ranks) / valid_queries
#     print(cmc_restuls)
#     return cmc_restuls


def match_matrix_test(args):
    encoder, decoder, discriminator = Experiment.load_model(args['model'], args['img_shape'], args['latent_size'], show_summary=False)
    transforms = Transforms.Compose([
        Transforms.ToTensor(),
        SlidingWindowTransformClass(args['img_shape'], args['img_shape'], 2),
    ])

    img_a = args['image']
    data_r = os.path.join(args['dataset'], '*', '*_*_R.jpg')
    data_r = glob.glob(data_r)
    data_l = os.path.join(args['dataset'], '*', '*_*_L.jpg')
    data_l = glob.glob(data_l)
    data = data_r + data_l
    data_len = len(data)
    logging.info('Calculating match matrix for {} images'.format(data_len))
    matrix = match_matrix(img_a, data, encoder, transforms)
    # matrix = match_matrix_precalc2()

    logging.info('Latentspace matrix is {}'.format(matrix))

    dataList = []
    # for n in range(data_len):
    #     logging.info('Image [{0: >58}] has lsd [{1:.18}]'.format(data[n], matrix[0, n]))
    #     dataList.append((matrix[0, n], data[n]))
    dataList = list(zip(matrix[0, :], data))

    logging.info('Sorted list\n')
    dataList.sort(key=lambda pair: pair[0])
    for n in range(data_len):
        logging.info('Image [{0: >58}] has lsd [{1:.18}]'.format(dataList[n][1], dataList[n][0]))


def cmc_test(args):
    encoder, decoder, discriminator = Experiment.load_model(args['model'], args['img_shape'], args['latent_size'], show_summary=False)
    transforms = Transforms.Compose([
        Transforms.ToTensor(),
        SlidingWindowTransformClass(args['img_shape'], args['img_shape'], 2),
    ])

    data_r = os.path.join(args['dataset'], '*', '*_*_R.jpg')
    data_r = glob.glob(data_r)
    data_l = os.path.join(args['dataset'], '*', '*_*_L.jpg')
    data_l = glob.glob(data_l)
    data = data_r + data_l
    data_len = len(data)
    logging.info('Calculating cmc for {} images'.format(data_len))
    distance = lambda target, image: calculate_patched_distance_batched(target, image, encoder, transforms)
    cmc_matrix = cmc(distance, data)
    logging.info('CMC matrix is {}'.format(cmc_matrix))


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')

    # Training settings
    parser = argparse.ArgumentParser(description='Impress latenspace search')
    parser.add_argument('image', type=str, metavar='IMG', help='jsonPath to target image')
    parser.add_argument('dataset', type=str, metavar='DATASET', help='jsonPath to datasetRoot')
    parser.add_argument('--model', default=None, required=True, type=str, metavar='PATH',
                        help='jsonPath to checkpoint of model')
    parser.add_argument('--latent_size', default=100, required=True, type=int, metavar='LATENT_SIZE',
                        help='latentspace size the model uses (default: 100)')
    parser.add_argument('--img_shape', default=256, required=True, type=int, metavar='IMG_SHAPE',
                        help='image size the model uses (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--gpuid', default='-1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args().__dict__

    args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()

    if args['cuda']:
        GPU.set(args['gpuid'], 400)
        cudnn.benchmark = True

    if 'model' not in args:
        raise AssertionError('a model to use must be passed')


if __name__ == '__main__':
    main()
