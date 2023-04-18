import argparse
import os, glob
import re
from preparation.utils import *


def get_directory(imgPath):
    match = re.search(r'/(\d{2,3})', imgPath)
    assert match, 'Could not get directory for "{}"'.format(imgPath)
    return re.sub(r'\d{2,3}[\w.]*', match.group(1), imgPath)


def get_img_part(imgPath):
    match = re.search(r'/(\d{2,3}[\w.]*)', imgPath)
    assert match, 'Could not get image part for "{}"'.format(imgPath)
    return match.group(1)


def move_and_replace(src, target_dir, target_entry, target, rep_target='.jpg'):
    target_entry = target_entry.replace(rep_target, target)
    target_source = src.replace(rep_target, target)
    os.rename(target_source, f'{target_dir}/{target_entry}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Impress dataset preparation')

    parser.add_argument('--path', default='~/datasets/registered', type=str, metavar='PATH',
                        help='path to dataset (default: ~/datasets/impress_aligned)')
    args = parser.parse_args()

    imagesPath = os.path.join(args.path, "*_*_merge.jpg")

    imagesPath = os.path.expanduser(imagesPath)

    # search files
    images = glob.glob(imagesPath)
    images.sort()

    length = len(images)
    printProgressBar(0, length, prefix='Progress:', suffix='Complete', length=50)
    for n, imgPath in enumerate(images):
        directory = get_directory(imgPath)
        if not os.path.exists(directory):
            os.mkdir(directory)

        img_part = get_img_part(imgPath)

        targets = ['.jpg', '.impression.jpg', '.impression.threshold.jpg', '.shoe.jpg', '.aligned.jpg', '.patches.json']
        for target in targets:
            move_and_replace(imgPath, directory, img_part, target)

        printProgressBar(n + 1, length, prefix='Progress:', suffix='Complete', length=50)


