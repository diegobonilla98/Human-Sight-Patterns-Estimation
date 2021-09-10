import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tqdm


IMAGES = '/media/bonilla/My Book/115_Paintings/hermitage'
MAPS = './paintings_maps'

images = sorted(glob.glob(os.path.join(IMAGES, '*.jpg')))
maps = sorted(glob.glob(os.path.join(MAPS, '*.jpg')))
with tqdm.tqdm(total=len(images)) as pbar:
    for i, m in zip(images, maps):
        ii = cv2.imread(i)
        mm = cv2.imread(m, 0)
        mm = np.uint8(np.where(mm > 200, 0, mm))
        mm = cv2.resize(mm, (ii.shape[1], ii.shape[0]), cv2.INTER_LANCZOS4)

        colormap = cv2.applyColorMap(mm, cv2.COLORMAP_JET)
        alpha = 0.6
        mix = cv2.addWeighted(ii, 1. - alpha, colormap, alpha, 1)

        cv2.imwrite(f'./paintings_maps/mixed/{i.split(os.sep)[-1]}', mix)
        pbar.update()
