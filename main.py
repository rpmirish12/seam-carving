import cv2
import numpy as np

import os
import errno

from os import path
from glob import glob

from seams1 import seam_carve

def main(image):
    output = seam_carve(image)
    cv2.imwrite("seam_removal.png", output.astype(np.uint8))


if __name__ == "__main__":
    seam_cut_image = cv2.imread('images/seam_carving_input.png', cv2.IMREAD_COLOR).astype(np.float_)
    main(seam_cut_image)