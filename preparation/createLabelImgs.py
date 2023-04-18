#!/usr/bin/python
#
# Converts the polygonal annotations of the Cityscapes dataset
# to data, where pixel values encode ground truth classes.
#
# With this tool, you can generate option
#   d) *labelTrainIds.png     : the class is encoded by its training ID
# This encoding might come handy for training purposes. You can use
# the file labes.py to define the training IDs that suit your needs.
# Note however, that once you submit or evaluate results, the regular
# IDs are needed.
#
# Uses the converter tool in 'json2labelImg.py'
# Uses the mapping defined in 'labels.py'
#

# python imports
import os, glob, sys

# impress imports
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from utils.helpers.csHelpers import printError
from utils.json2labelImg import json2labelImg


# The main method
def main():
    # Where to look for Impress
    if 'IMPRESS_DATASET' in os.environ:
        impressPath = os.environ['IMPRESS_DATASET']
    else:
        impressPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    # how to search for all ground truth
    searchFine = os.path.join(impressPath, "*", "*_*_*.jpg_*_*_*.patches.json")

    # search files
    files = glob.glob(searchFine)
    files.sort()

    # quit if we did not find anything
    if not files:
        printError("Did not find any files. Please consult the README.")

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format(progress * 100 / len(files)), end=' ')
    for f in files:
        # create the output filename
        dst = f.replace(".patches.json", ".categoryIds.png")

        # do the conversion
        try:
            json2labelImg(f, dst, "color")
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format(progress * 100 / len(files)), end=' ')
        sys.stdout.flush()


# call the main
if __name__ == "__main__":
    main()
