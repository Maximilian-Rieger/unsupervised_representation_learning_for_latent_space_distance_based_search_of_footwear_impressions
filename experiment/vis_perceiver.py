import argparse
import math
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
from einops import rearrange, repeat
from tqdm import tqdm
import numpy as np
from latentspace_distance import load_image

from torchvision.utils import save_image, make_grid


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
    parser.add_argument('--model', default=None, required=True, type=str, metavar='MODEL',
                        help='name of model to use')
    parser.add_argument('--model-weights', default=None, required=True, type=str, metavar='WEIGHT_PATH',
                        help='jsonPath to checkpoint of model')
    parser.add_argument('--config', required=True, type=str, metavar='CONFIG', nargs='+',
                        help='path to config file with model config')
    parser.add_argument('--image', default=None, required=False, type=str, metavar='IMAGE_PATH',
                        help='path to image to be reconstructed')
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
    batchsize = configArgs['batchsize']
    out_shape = [batchsize, 256, 256, 1]

    perceiver, deceiver, _ = Experiment.load_model(modelWeights, batchsize, **configArgs['model'])
    perceiver.eval()
    deceiver.eval()


    try:
        os.mkdir(os.path.join(modelWeights, 'latents'))
    except FileExistsError:
        pass
    with torch.no_grad():
        pbar = tqdm(range(perceiver.num_latents))
        pbar.set_description('Vector Vis:')
        z = torch.zeros((batchsize, perceiver.num_latents, perceiver.latent_dim)).to(GPU.device)
        ones = torch.ones(perceiver.latent_dim).to(GPU.device)
        for vec in pbar:
            latent = z.detach()
            latent[:, 0:vec+1] = perceiver.latents[0:vec+1]
            latents_imgs = deceiver(latent, out_shape)
            latents_imgs = rearrange(latents_imgs, 'b h w c -> b c h w')
            save_image(latents_imgs[0], os.path.join(modelWeights, 'latents', f'latent_{vec}_vis.png'))
            del latents_imgs
        logging.info('Single Vector vis finished')

        latents = repeat(perceiver.latents, 'n d -> b n d', b=batchsize)
        latents_imgs, latents_imgs_stages = deceiver.forward_with_staged_data(latents, out_shape)
        latents_imgs = rearrange(latents_imgs, 'b h w c -> b c h w')
        grid = make_grid(latents_imgs, nrow=(batchsize // 2))
        save_image(grid, os.path.join(modelWeights, 'latents', 'perceiver_latents_vis.png'))
        latents_imgs_stages = [rearrange(latent_img, 'b h w c -> b c h w')[0] for latent_img in latents_imgs_stages]
        for i, latent_img in enumerate(latents_imgs_stages):
            grid = make_grid(latent_img, nrow=(batchsize // 2))
            save_image(grid, os.path.join(modelWeights, 'latents', f'perceiver_latents_vis_stage_{i}.png'))

        latents = repeat(deceiver.latents, 'n d -> b n d', b=batchsize)
        latents_imgs, latents_imgs_stages = deceiver.forward_with_staged_data(latents, out_shape)
        latents_imgs = rearrange(latents_imgs, 'b h w c -> b c h w')
        grid = make_grid(latents_imgs, nrow=(batchsize // 2))
        save_image(grid, os.path.join(modelWeights, 'latents', 'deceiver_latents_vis_complete.png'))
        latents_imgs_stages = [rearrange(latent_img, 'b h w c -> b c h w')[0] for latent_img in latents_imgs_stages]
        for i, latent_img in enumerate(latents_imgs_stages):
            grid = make_grid(latent_img, nrow=(batchsize // 2))
            save_image(grid, os.path.join(modelWeights, 'latents', f'deceiver_latents_vis_stage_{i}.png'))

        logging.info('Latents vis finished')

        if "image" in args:
            transforms = Transforms.Compose([
                Transforms.Resize((256, 256)),
                Transforms.Grayscale(),
                Transforms.ToTensor(),
                Transforms.Lambda(lambd=lambda x: rearrange(x, 'c h w -> h w c'))
            ])
            img = load_image(args['image'], transforms)
            img = repeat(img, 'h w c -> b h w c', b=batchsize)
            img = img.to(GPU.device)
            enc_img, perceiver_attn_maps = perceiver.forward_with_attention_maps(img)
            # for index, attn_map in enumerate(perceiver_attn_maps):
            #     # attn_grid = make_grid(attn_map.view(batch_size, perceiver.num_latents, *img_shape[1:3]))
            #     attn_map = 1 - attn_map
            #     # rearrange(attn_map, ' b l (h w) -> b l h w', h=img_shape[1])
            #     attn_grid = make_grid(make_grid(attn_map.view(batchsize, perceiver.num_latents, *out_shape[1:3]), pad_value=1).unsqueeze(1), pad_value=0.5)
            #     save_image(attn_grid.data, os.path.join(modelWeights, 'latents', f'perceiver_attention_{index}.png'), nrow=(batchsize // 2))

            pbar = tqdm(range(perceiver.num_latents))
            pbar.set_description('Encoding info vis:')
            z = torch.zeros((batchsize, perceiver.num_latents, perceiver.latent_dim)).to(GPU.device)
            ones = torch.ones(perceiver.latent_dim).to(GPU.device)
            for vec in pbar:
                latent = z.detach()
                latent[:, 0:vec+1] = enc_img[0, 0:vec+1]
                latents_imgs = deceiver(latent, out_shape)
                latents_imgs = rearrange(latents_imgs, 'b h w c -> b c h w')
                save_image(latents_imgs[0], os.path.join(modelWeights, 'latents', f'culled_encoding_{vec}_vis.png'))
                del latents_imgs
            logging.info('Culled encoding vis finished')


if __name__ == '__main__':
    main()
