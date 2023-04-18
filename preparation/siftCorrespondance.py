import os, glob, sys
from PIL import Image
import logging
import argparse
import numpy as np

import cv2

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s - %(message)s')


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def rotate(img, angle):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    return img


def pad(img, size):
    assert size >= img.shape[0] and size >= img.shape[1],\
        'pad size must be bigger than width and height, size: {0} width: {1} height: {2}'\
        .format(size, img.shape[0], img.shape[1])
    top = int((size - img.shape[0]) / 2.0)  # shape[0] = rows
    bottom = top
    left = int((size - img.shape[1]) / 2.0)  # shape[0] = rows
    right = left
    borderType = cv2.BORDER_CONSTANT
    img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, [232, 58, 19])
    return img


def crop(img, height=None, width=None):
    if width is None and height is None:
        return img
    img_height, img_width = img.shape[0:2]
    width = img_width if width is None else width
    height = img_height if height is None else height

    diff_y = (img_height - height) // 2
    diff_x = (img_width - width) // 2
    return img[diff_y:(diff_y + height), diff_x:(diff_x + width)]


def flipv(img):
    height, width = img.shape[:2]
    img2 = np.zeros([height, width, 3], np.uint8)
    for i in range(height):
        img2[i, :] = img[height-i-1, :]
    return img2


def fliph(img):
    height, width = img.shape[:2]
    img2 = np.zeros([height, width, 3], np.uint8)
    for i in range(width):
        img2[:, i] = img[:, width-i-1]
    return img2


def siftKeyPoints(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    logging.info('convert data to grayscale')

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    logging.info('detected keypoints')
    return kp1, kp2, des1, des2


def preProcess(img1, img2, width):
    img1 = image_resize(img1, width=width)
    img2 = image_resize(img2, width=width)
    pass



def mainBFMatch(img1, img2, kp1, kp2, des1, des2, nmatches=10):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    logging.info('calculated matches')

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    logging.info('sorted matches')

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:nmatches], None, flags=2)
    logging.info('drawn matches')
    return img3


def mainFLANMatch(img1, img2, kp1, kp2, des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    logging.info('calculated matches')

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    logging.info('sorted matches')

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    logging.info('drawn matches')

    return img3

def homography(img1, img2, kp1, kp2, des1, des2, MIN_MATCH_COUNT=10):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print
        "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    logging.info('drawn matches')

    return img3


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):
    # Convert data to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def mainBlob(img, params=None):
    if isinstance(img, str):
        img = cv2.imread(img)

    if isinstance(params, dict):
        if params:
            _params = cv2.SimpleBlobDetector_Params()
            if 'treshhold' in params:
                _params.minThreshold = params['treshhold']['start']
                _params.maxThreshold = params['treshhold']['end']
            if 'area' in params:
                _params.filterByArea = True
                _params.minArea = params['area']['start']
                _params.maxArea = params['area']['end']
            if 'inertia' in params:
                _params.filterByInertia = True
                _params.minInertiaRatio = params['inertia']['start']
                _params.maxInertiaRatio = params['inertia']['end']
            params = _params
        else:
            params = None

    if params is None:
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 255

        # params.filterByArea = True
        # params.minArea = 1500

        params.filterByInertia = True
        params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)

    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_with_keypoints



if __name__ == "__main__":
    boot = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress schuhe+spezial/001/001_01_2.jpg'
    # boot = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress/1/1_3_L.jpg'
    sole = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress/1/1_1_R.jpg'
    # sole2 = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress/1/1_3_L.jpg'
    # sole3 = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress/1/1_a_R.jpg'

    boot_img = cv2.imread(boot)
    sole_img = cv2.imread(sole)
    # sole_img2 = cv2.imread(sole2)
    # sole_img3 = cv2.imread(sole3)
    logging.info('read data')

    boot_img = image_resize(boot_img, width=512)
    sole_img = image_resize(sole_img, width=512)
    # sole_img2 = image_resize(sole_img2, width=512)
    # sole_img3 = image_resize(sole_img3, width=512)
    logging.info('resized data')

    # boot_img = fliph(boot_img)
    # sole_img2 = fliph(sole_img2)
    alpha = 0.5

    # cv2.addWeighted(sole_img2, alpha, sole_img, 1, 0, sole_img)
    # cv2.addWeighted(sole_img3, alpha, sole_img, 1, 0, sole_img)
    # cv2.addWeighted(sole_img2, alpha, sole_img, 1 - alpha, 0, sole_img)

    boot_img = crop(boot_img, height=220)

    boot_img = pad(boot_img, 512)
    # sole_img = pad(sole_img, 512)

    boot_img = rotate(boot_img, -90)
    # sole_img = rotate(sole_img, 90)

    boot_img = crop(boot_img, width=220)
    sole_img = crop(sole_img, width=365)

    boot_img = image_resize(boot_img, width=512)
    sole_img = image_resize(sole_img, width=512)

    boot_img_gray = cv2.cvtColor(boot_img, cv2.COLOR_RGB2GRAY)
    sole_img_gray = cv2.cvtColor(sole_img, cv2.COLOR_RGB2GRAY)
    logging.info('convert data to grayscale')

    sift = cv2.xfeatures2d.SURF_create()
    boot_img_kp, boot_img_des = sift.detectAndCompute(boot_img_gray, None)
    sole_img_kp, sole_img_des = sift.detectAndCompute(sole_img_gray, None)
    logging.info('detected keypoints')

    # img3 = mainBFMatch(boot_img, sole_img, boot_img_kp, sole_img_kp, boot_img_des, sole_img_des)
    # img3 = mainFLANMatch(boot_img, sole_img, boot_img_kp, sole_img_kp, boot_img_des, sole_img_des)
    img3 = homography(boot_img, sole_img, boot_img_kp, sole_img_kp, boot_img_des, sole_img_des)

    cv2.imwrite(boot.replace('jpg', 'sift.flan.hom.cor.jpg'), img3)

if __name__ == "_g_main__":
    boot = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress schuhe+spezial/001/001_01_2.jpg'
    # boot = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress/1/1_3_L.jpg'
    sole = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress/1/1_1_R.jpg'
    # sole2 = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress/1/1_3_L.jpg'
    # sole3 = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress/1/1_a_R.jpg'

    boot_img = cv2.imread(boot)
    sole_img = cv2.imread(sole)
    # sole_img2 = cv2.imread(sole2)
    # sole_img3 = cv2.imread(sole3)
    logging.info('read data')

    boot_img = image_resize(boot_img, width=512)
    sole_img = image_resize(sole_img, width=512)
    # sole_img2 = image_resize(sole_img2, width=512)
    # sole_img3 = image_resize(sole_img3, width=512)
    logging.info('resized data')

    # boot_img = fliph(boot_img)
    # sole_img2 = fliph(sole_img2)
    alpha = 0.5

    boot_img = crop(boot_img, height=220)

    boot_img = pad(boot_img, 512)
    # sole_img = pad(sole_img, 512)

    boot_img = rotate(boot_img, -90)
    # sole_img = rotate(sole_img, 90)

    boot_img = crop(boot_img, width=220)
    sole_img = crop(sole_img, width=365)

    boot_img = image_resize(boot_img, width=512)
    sole_img = image_resize(sole_img, width=512)

    boot_img_gray = cv2.cvtColor(boot_img, cv2.COLOR_RGB2GRAY)
    sole_img_gray = cv2.cvtColor(sole_img, cv2.COLOR_RGB2GRAY)
    logging.info('convert data to grayscale')

    logging.info('detected keypoints')

    img3 = mainBlob(boot_img)
    cv2.imwrite(boot.replace('jpg', 'sift.flan.hom.cor.jpg'), img3)






if __name__ == '__fmain__':
    boot = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress schuhe+spezial/001/001_01_2.jpg'

    sole = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress/1/1_1_R.jpg'


    # Read reference image
    refFilename = sole
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = boot
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning data ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)