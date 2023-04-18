import argparse
import torch
import torch.nn.init
import torch.backends.cudnn as cudnn
import os
import logging
from utils.utils import GPU
from utils.utils import load_config
from utils.utils import dynamic_import_experiment


experiments = [
    'experiment_aae',
    'experiment_aae_noise',
    'experiment_patches',
    'experiment_resize',
    'experiment_ae',
    'experiment_dcgan',
    'experiment_gan',
    'experiment_gan2',
    'experiment_resunet',
    'experiment_simple_gan',
    'experiment_suprvae',
    'experiment_suprvae_patches',
    'experiment_suprvae_gan',
    'experiment_suprvae_cl',
    'experiment_betavae',
    'experiment_unet',
    'experiment_vae_vampprior',
    'experiment_vqvae',
    'experiment_vqvae_dis',
    'experiment_vqvae_slp',
    'experiment_vqvae_gd',
    'experiment_vqvae_transformer',
    'experiment_perceiver',
    'experiment_perceiver_at',
    'experiment_perceiver_at_latent_space',
    'experiment_perceiver_at_internal',
    'experiment_perceiver_autoreg',
    'experiment_perceiver_autoreg_contrast',
    'experiment_perceiver_som_loss',
    'experiment_perceiver_relt_som_loss',
    'experiment_perceiver_deaugmenting',
    'experiment_perceiver_msls',
    'experiment_perceiver_vlt',
    'experiment_perceiver_cl',
    'experiment_perceiver_wae',
    'experiment_vit',
    'experiment_vicreg',
    'experiment_perceiver_mnist',
]


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')

    # Training settings
    parser = argparse.ArgumentParser(description='Impress reconstruction')

    parser.add_argument('-c', '--config', nargs='+', required=True)
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='jsonPath to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpuid', default='-1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('-w', '--workers', default=-1, type=int,
                        help='Count of max Workers to use')
    parser.add_argument('-m', '--mode', default='experiment_suprvae', required=True, type=str, choices=experiments)
    args = load_config(parser.parse_args())

    args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()
    args['log_dir'] = os.path.expanduser(args['log_dir'])

    if not os.path.exists(args['log_dir']):
        os.makedirs(args['log_dir'])

    if args['cuda']:
        GPU.set(args['gpuid'], 400)
        cudnn.benchmark = True

    if not args['workers'] == -1:
        args['training']['worker'] = args['workers']
        args['validation']['worker'] = args['workers']
    del args['workers']

    if 'mode' not in args:
        raise AssertionError('a mode must be passed')
    mode = args['mode']
    del args['mode']

    if mode not in experiments:
        raise AssertionError('Mode must be an experiment package')
    experiment = dynamic_import_experiment(mode, args)
    try:
        experiment.run()
    except Exception as Arguments:
        logging.exception('Something went wrong in the experiment')


if __name__ == '__main__':
    main()
