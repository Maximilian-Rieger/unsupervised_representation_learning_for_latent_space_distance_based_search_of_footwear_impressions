import argparse
import os, glob
from preparation.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Impress dataset preparation')

    parser.add_argument('--path', default='~/datasets/registered', type=str, metavar='PATH',
                        help='path to dataset (default: ~/datasets/impress_aligned)')
    args = parser.parse_args()

    annotationsPath = os.path.join(args.path, "*_*_merge.patches.json")
    imagesPath = os.path.join(args.path, "*_*_merge.jpg")

    annotationsPath = os.path.expanduser(annotationsPath)
    imagesPath = os.path.expanduser(imagesPath)

    # search files
    annotations = glob.glob(annotationsPath)
    annotations.sort()
    images = glob.glob(imagesPath)
    images.sort()

    items = list(zip(annotations, images))
    length = len(items)
    printProgressBar(0, length, prefix='Progress:', suffix='Complete', length=50)
    for n, pair in enumerate(items):
        annotationPath, imgPath = pair
        impressionPoints, shoePoints = parsePoints(annotationPath)

        img = cv2.imread(imgPath)
        impression, shoe = split(img)

        shoePoints = [[x - shoe.shape[1], y] for x, y in shoePoints]
        # pair points with their nearest corresponding points
        # impressionPoints, shoePoints = pairPoints_(impressionPoints, shoePoints)
        # impressionPoints, shoePoints = trim(impressionPoints, shoePoints)

        # convert points to numpy array for OpenCV
        impressionPoints, shoePoints = np.float32(impressionPoints), np.float32(shoePoints)

        # calculate and transform affine from moved points
        impression = affineTransfrom(impression, impressionPoints, shoePoints, imgPath)

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

        printProgressBar(n + 1, length, prefix='Progress:', suffix='Complete', length=50)


