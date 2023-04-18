import argparse
import os, glob
import re
import cv2
from preparation.utils import *


def get_directory(imgPath):
    match = re.search(r'/(\d{1,3})', imgPath)
    assert match, 'Could not get directory for "{}"'.format(imgPath)
    return re.sub(r'\d{1,3}[\w.]+', match.group(1), imgPath)


def get_img_part(imgPath):
    match = re.search(r'/(\d{1,3}[\w.]+)', imgPath)
    assert match, 'Could not get image part for "{}"'.format(imgPath)
    return match.group(1)


def move_and_replace(src, target_dir, target_entry, target, rep_target='.jpg'):
    target_entry = target_entry.replace(rep_target, target)
    target_source = src.replace(rep_target, target)
    os.rename(target_source, f'{target_dir}/{target_entry}')


def save_img(img, target_dir, target_entry, target, rep_target='.jpg'):
    target_entry = target_entry.replace(rep_target, target)
    cv2.imwrite(f'{target_dir}/{target_entry}', img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Impress dataset preparation')

    parser.add_argument('--path', default='~/datasets/tmp2', type=str, metavar='PATH',
                        help='path to dataset (default: ~/datasets/impress_aligned)')
    args = parser.parse_args()

    imagesPath = os.path.join(args.path, "*_*_merge.aligned.jpg")

    imagesPath = os.path.expanduser(imagesPath)

    # search files
    images = glob.glob(imagesPath)
    images.sort()

    length = len(images)
    printProgressBar(0, length, prefix='Progress:', suffix='Complete', length=50)
    for n, imgPath in enumerate(images):
        impression, shoe = split(cv2.imread(imgPath))
        directory = get_directory(imgPath)
        if not os.path.exists(directory):
            os.mkdir(directory)

        img_part = get_img_part(imgPath)

        save_img(impression, directory, img_part, '.impression.jpg', '.aligned.jpg')

        impression_threshold = impression
        impression_threshold[impression_threshold <= 127] = 1
        impression_threshold[impression_threshold > 127] = 0
        save_img(impression_threshold, directory, img_part, '.impression.threshold.jpg', '.aligned.jpg')

        save_img(shoe, directory, img_part, '.shoe.jpg', '.aligned.jpg')
        move_and_replace(imgPath, directory, img_part, '.jpg')

        printProgressBar(n + 1, length, prefix='Progress:', suffix='Complete', length=50)


