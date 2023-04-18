import cv2
import logging
import json
import numpy as np
from preparation.utils import *


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s - %(message)s')

logging.info(f'OpenCV version: {cv2.__version__}')
# logging.info(f'OpenCV modules: {dir(cv2)}')

jsonPath = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress_registered/13/13_L_merge.patches.json'
imgPath = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress_registered/13/13_L_merge.jpg'



if __name__ == '__main__':
    impressionPoints = []
    shoePoints = []
    with open(jsonPath, 'r') as file:
        file = json.load(file)
        impression, shoe = file
        for point in impression['polygon']['points']:
            impressionPoints += [point]
        for point in shoe['polygon']['points']:
            shoePoints += [point]

    img = cv2.imread(imgPath)
    impression, shoe = split(img)

    shoePoints = [[x - shoe.shape[1], y] for x, y in shoePoints]

    # pair points with their nearest corresponding points
    impressionPoints, shoePoints = pairPoints_(impressionPoints, shoePoints)

    # mark points for visual inspection of point pairs
    markPoints(impression, shoe, impressionPoints, shoePoints)

    # convert points to numpy array for OpenCV
    impressionPoints, shoePoints = np.float32(impressionPoints), np.float32(shoePoints)

    # calculate and transform affine from moved points
    impressionTransformed = affineTransfrom(impression, impressionPoints, shoePoints)

    imgNew = merge(impressionTransformed, shoe)
    cv2.imwrite("impression.jpg", impression)
    cv2.imwrite("impressionTransformed.jpg", impressionTransformed)
    cv2.imwrite("shoe.jpg", shoe)
    cv2.imwrite("merged.jpg", imgNew)
    logging.info('Finished')





