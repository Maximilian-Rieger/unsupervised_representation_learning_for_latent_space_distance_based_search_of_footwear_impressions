import cv2
import numpy as np
import json
import sys
import math
import os
from PIL import Image
import glob
from tqdm import tqdm


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s (%s / %s)' % (prefix, bar, percent, suffix, iteration, total), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()
    sys.stdout.flush()


def pairPoints2(src_pts, dst_pts):
    assert len(src_pts) == len(dst_pts), 'Number of points in each image has to match! {} != {}'\
        .format(len(src_pts), len(dst_pts))
    pts = []
    for src_point in src_pts:
        min_dist = 10000000
        min_index = -1
        for i, dst_point in enumerate(dst_pts):
            dist = ((dst_point[0] - src_point[0]) ** 2 + (dst_point[1] - src_point[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                min_index = i
        pts += [dst_pts.pop(min_index)]
    return src_pts, pts


def pairPoints(src_pts, dst_pts):
    pts = []
    used = []
    for src_point in src_pts:
        min_dist = 10000000
        min_index = -1
        for i, dst_point in enumerate(dst_pts):
            dist = ((dst_point[0] - src_point[0]) ** 2 + (dst_point[1] - src_point[1]) ** 2) ** 0.5
            if dist < min_dist and i not in used:
                min_dist = dist
                min_index = i
        pts += [dst_pts[min_index]]
        used += [min_index]
    return src_pts, pts


def clockwiseangle_and_distance(points, refvec=[0, 1]):
    origin = np.average(points, 0)
    refvec = np.multiply(origin, refvec)

    def sort_clockwise(point):
        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector
    return sort_clockwise


# def pairPoints_(src_pts, dst_pts):
#     src_pts = sorted(src_pts, key=clockwiseangle_and_distance(src_pts))
#     dst_pts = sorted(dst_pts, key=clockwiseangle_and_distance(src_pts))
#     return src_pts, dst_pts


def pairPoints_(src_pts, dst_pts):
    src_pts = sorted(src_pts, key=lambda k: [k[1], k[0]])
    dst_pts = sorted(dst_pts, key=lambda k: [k[1], k[0]])
    return src_pts, dst_pts


def checkPoints(img, pts, affine, max_dist=0.01, mark_points=False):
    rows, cols, channels = img.shape
    img_center = (cols * 0.5, rows * 0.5)
    affine = np.stack([affine[0], affine[1], [0, 0, 1]])
    pts = [np.matmul(affine, np.stack([pt[0], pt[1], 1]))[:2] for pt in pts]
    center = np.average(pts, axis=0)
    in_center = inCircle(center, img_center, ((rows + cols) * 0.5) * max_dist)
    return in_center


def inCircle(point, center, radius):
    x, y = point
    center_x, center_y = center
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2


def trim(src_pts, dst_pts):
    length = min(len(src_pts), len(dst_pts))
    return src_pts[:length], dst_pts[:length]


def affineTransfrom(img, src_pts, dst_pts, imgPath):
    rows, cols, ch = img.shape
    # translation, matrix = compute_affine_transform(dst_pts, src_pts)
    # matrix = np.column_stack([matrix, translation])
    matrix, inliners = cv2.estimateAffine2D(src_pts, dst_pts)
    result = cv2.warpAffine(img, matrix, (cols, rows), borderValue=(255, 255, 255))
    if not checkPoints(img, src_pts, matrix, 0.1):
        # cv2.circle(result, (int(cols * 0.5), int(rows * 0.5)), int(((rows + cols) * 0.5) * 0.1), (0, 0, 0), -1)
        # markPoints(result, src_pts)
        print('Transformed centroid not in center! {}'.format(imgPath))
    return result


def get_affine_transfrom(src_pts, dst_pts):
    matrix, inliners = cv2.estimateAffine2D(src_pts, dst_pts)
    return matrix


def transform_affine(img, affine, borderValue=(255, 255, 255)):
    rows, cols, ch = img.shape
    result = cv2.warpAffine(img, affine, (cols, rows), borderValue=borderValue)
    return result


def transform_points_affine(pts, affine):
    affine = np.stack([affine[0], affine[1], [0, 0, 1]])
    pts = [np.matmul(affine, np.stack([pt[0], pt[1], 1]))[:2] for pt in pts]
    return pts


def split(img):
    height, width = img.shape[:2]
    start_row, start_col = 0, 0
    # Let's get the ending pixel coordinates (bottom right of cropped top)
    end_row, end_col = height, int(width * 0.5)
    cropped_left = img[start_row:end_row, start_col:end_col]
    start_col, end_col = end_col, width
    cropped_right = img[start_row:end_row, start_col:end_col]
    return cropped_left, cropped_right


def merge(img1, img2):
    assert img1.shape[0] == img2.shape[0]
    img = np.column_stack([img1, img2])
    return img


def markPoint(img, x, y, text=None, circle_color=(0, 0, 0), circle_size=100, text_color=(255, 255, 255), font_size=10, font_width=10, circle_fill=-1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.circle(img, (int(x), int(y)), circle_size, circle_color, circle_fill)
    if text is not None:
        cv2.putText(img, str(text), (int(x), int(y)), font, font_size, text_color, font_width, cv2.LINE_AA)


def markPoints(img, pts, invert=False):
    circle_color = (0, 0, 255) if not invert else (255, 0, 0)
    for i, point in enumerate(pts):
        markPoint(img, text=i, *point, circle_color=(0, 0, 255), text_color=(0, 255, 0))


def markPointsDouble(img1, img2, pts1, pts2):
    markPoints(img1, pts1)
    markPoints(img2, pts1, invert=True)
    markPoints(img1, pts2, invert=True)
    markPoints(img2, pts2)


def parsePoints(annotation):
    impressionPoints = []
    shoePoints = []
    with open(annotation, 'r') as file:
        file = json.load(file)
        impression, shoe = file
        for point in impression['polygon']['points']:
            impressionPoints += [point]
        for point in shoe['polygon']['points']:
            shoePoints += [point]
    if np.average(impressionPoints, axis=0)[0] > np.average(shoePoints, axis=0)[0]:
        print('swapped points {}'.format(annotation))
        impressionPoints, shoePoints = shoePoints, impressionPoints
    return impressionPoints, shoePoints


def savePoints(annotation, impressionPoints, shoePoints):
    with open(annotation, 'w') as file:
        jsonStruct = [
            {
                'polygon': {
                    'points': impressionPoints
                }
            },
            {
                'polygon': {
                    'points': shoePoints
                }
            }
        ]
        json.dump(jsonStruct, file)


def filter_by_mean(directory, min, max, image_def='patch_*.png', sub_dir='training', filter_dir='exclude'):
    """
    Filter image data and extrtact im
    @params:
        directory   - Required  : path to dataset (Str)
        min         - Required  : minimum value [exclusive] (Float)
        max         - Required  : maximum value [exclusive] (Float)
        image_def   - Optional  : glob pattern for image files (Str)
        sub_dir     - Optional  : sub directory in dataset dir where the image files are located (Str)
        filter_dir  - Optional  : sub directory in dataset dir where the filtered images will be moved to (Str)
    """
    directory = os.path.expanduser(directory)
    if not os.path.exists(os.path.join(directory, filter_dir)):
        os.mkdir(os.path.join(directory, filter_dir), )
    pattern = os.path.join(directory, sub_dir, image_def)
    patches = glob.glob(pattern)
    pbar = tqdm(patches, 'Filtering')
    filtered = 0
    for patch in pbar:
        img_arr = np.array(Image.open(patch))
        mean = img_arr.mean()
        if mean < min or mean > max:
            os.rename(patch, patch.replace(sub_dir, filter_dir))
            filtered += 1
    return filtered


def append_name(directory, suffix):
     directory = os.path.expanduser(directory)
     pattern = os.path.join(directory, 'training', 'patch_*.png')
     patches = glob.glob(pattern)
     pbar = tqdm(patches, 'Renaming')
     for patch in pbar:
         os.rename(patch, patch.replace('.png', suffix + '.png'))