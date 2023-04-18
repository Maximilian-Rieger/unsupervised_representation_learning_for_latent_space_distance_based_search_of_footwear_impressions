import argparse
import os, glob
from tqdm import tqdm


import numpy as np
import cv2

from preparation.utils import *


def fixSeperator(path, current_seperator, new_seperator):
    return path.replace(current_seperator, new_seperator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Impress dataset preparation')

    parser.add_argument('--path', default='~/datasets/registered_full', type=str, metavar='PATH',
                        help='path to dataset (default: ~/datasets/registered)')
    parser.add_argument('--patch', default='*_*_merge.patches.json', type=str, metavar='PATCH',
                        help='glob expression to find image annotations')
    parser.add_argument('--image', default='*_*_merge.jpg', type=str, metavar='IMAGE',
                        help='glob expression to find images')
    args = parser.parse_args()

    annotationsPath = os.path.join(args.path, "*_*_merge.patches.json")
    imagesPath = os.path.join(args.path, "*_*_merge.jpg")

    annotationsPath = os.path.expanduser(annotationsPath)
    imagesPath = os.path.expanduser(imagesPath)

    print(annotationsPath)
    print(imagesPath)

    # search files
    annotations = glob.glob(annotationsPath)
    annotations.sort()
    images = glob.glob(imagesPath)
    images.sort()

    assert len(images) > 0, 'No images found'

    items = list(zip(annotations, images))
    pbar = tqdm(items, desc='Progress:')
    for n, pair in enumerate(pbar):
        annotationPath, imgPath = pair
        impressionPoints, shoePoints = parsePoints(annotationPath)
        img = cv2.imread(imgPath)
        impression, shoe = split(img)

        shoePoints = [[x - shoe.shape[1], y] for x, y in shoePoints]
        # pair points with their nearest corresponding points
        # impressionPoints, shoePoints = pairPoints_(impressionPoints, shoePoints)
        impressionPoints, shoePoints = trim(impressionPoints, shoePoints)

        # convert points to numpy array for OpenCV
        impressionPoints, shoePoints = np.float32(impressionPoints), np.float32(shoePoints)

        # calculate and transform affine from moved points
        affine = get_affine_transfrom(shoePoints, impressionPoints)
        shoe = transform_affine(shoe, affine)
        # img_center = (shoe[0] * 0.5, shoe[1] * 0.5)
        # pts = transform_points_affine(shoePoints, affine)
        # if not inCircle(np.average(pts, axis=0), (impression[0] * 0.5, impression[1] * 0.5), 0.1):
        #     print('Transformed centroid not in center! {}'.format(imgPath))
        #     print()
        #     printProgressBar(n + 1, length, prefix='Progress:', suffix='Complete', length=50)
        #     continue

        impressionPath = imgPath.replace('.jpg', '.impression.jpg')
        impressionTresholdedPath = imgPath.replace('.jpg', '.impression.threshold.jpg')
        shoePath = imgPath.replace('.jpg', '.shoe.jpg')

        impression_threshold = impression.copy()
        # impression_threshold = (impression <= 127) * impression
        impression_threshold[impression_threshold <= 127] = 1
        impression_threshold[impression_threshold > 127] = 0

        cv2.imwrite(impressionPath, impression)
        cv2.imwrite(impressionTresholdedPath, impression_threshold)
        cv2.imwrite(shoePath, shoe)

        # mark points for visual inspection of point pairs
        # markPointsDouble(impression, shoe, impressionPoints, shoePoints)
        imgNew = merge(impression, shoe)

        cv2.imwrite(imgPath.replace('.jpg', '.aligned.jpg'), imgNew)


