import os, glob
import cv2
import numpy as np
import re
from preparation.utils import *


def rot90n(img, n):
    for x in range(n):
        img = np.rot90(img)
    return img


def rot270(img):
    return rot90n(img, 3)


def rot180(img):
    return rot90n(img, 2)


def get_impression(shoe, target):
    impression = shoe.replace('schuhe+spezial', 'sorted-inkless')
    impression = re.sub(r"/(0?(\d+))", r"/\1/\2", impression) + target

    impression_img = cv2.imread(impression)
    impression_img = rot180(impression_img)
    # impression_img = np.flip(impression_img, axis=1)
    return impression_img, impression


def get_entry(impression, target):
    entry = impression.replace('/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/sorted-inkless', '')
    entry = re.sub(r"/\d+/", "", entry)
    entry = entry.replace('.jpg', target + '_merge.jpg')
    return entry


def get_shoe(shoe, target):
    shoe = re.sub(r"/(\d+)", r"/\1/\1", shoe) + target

    shoe_img = cv2.imread(shoe)
    shoe_img = rot270(shoe_img)
    shoe_img = np.flip(shoe_img, axis=1)

    height = int(shoe_img.shape[0] * (5100 / shoe_img.shape[1]))
    shoe_img = cv2.resize(shoe_img, (5100, height), interpolation=cv2.INTER_AREA)
    padded = np.zeros((8400, 5100, 3))
    padded[:shoe_img.shape[0], :shoe_img.shape[1]] = shoe_img
    return padded


if __name__ == '__main__':
    base_path = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/schuhe+spezial'

    # shoes = os.path.join(base_path, "*", '*_*_2.jpg')
    shoes = os.path.join(base_path, "*")
    shoes = glob.glob(shoes)
    length = len(shoes)

    printProgressBar(0, length, prefix='Progress:', suffix='Complete', length=80)
    for n, shoe in enumerate(shoes):
        impression = ''
        # if n < 155: continue
        # if n < 52: continue
        # if n > 53 and n < 107: continue
        # if n > 113 and n < 172: continue
        # if n > 179 and n < 189: continue
        try:
            for targets in [('_01_2.jpg', '_3_3.jpg', '_L'), ('_01_3.jpg', '_3_1.jpg', '_R')]:
                shoe_img = get_shoe(shoe, targets[0])
                impression_img, impression = get_impression(shoe, targets[1])

                combined = np.concatenate((impression_img, shoe_img), axis=1)
                del impression_img, shoe_img
                entry = get_entry(impression, targets[2])
                cv2.imwrite(f"/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/combined/{entry}", combined)
                del combined

        except ValueError as e:
            print(f"An error has occured!: [{n}] {shoe}\n{impression}\n{e}")
            # if 'shoe_img' in dir() and 'impression_img' in dir():
            #     print(f"{shoe_img.shape} {impression_img.shape}\n")
            continue
        printProgressBar(n + 1, length, prefix='Progress:', suffix='Complete', length=80)


