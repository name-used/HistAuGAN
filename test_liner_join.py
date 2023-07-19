import json
import random

import torch
import cv2
import numpy as np
from augmentations import generate_hist_augs, opts, mean_domains, std_domains
from histaugan.model import MD_multi
import imageio
from torchvision import transforms
import time

from utils import PPlot


def main():

    with open(rf'D:\jassorRepository\OCELOT_Dataset\jassor\tissue\image_transforms\meta.json') as f:
        meta = json.load(f)

    codes = [str(index).zfill(3) for index in range(1, 401)]
    random.shuffle(codes)

    # 展图
    for code in codes:
        pplt = PPlot(code)
        for d in range(5):
            for za in range(6):
                image = cv2.imread(rf'D:\jassorRepository\OCELOT_Dataset\jassor\tissue\image_transforms\{d}_{za}\{code}.jpg')
                pplt.add(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pplt.show()


main()
