import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from BoniDL.utils import allow_gpu_growth
from tensorflow.keras.models import load_model
import tqdm


allow_gpu_growth()
model = load_model('./model.h5')


def load_image(p):
    i = cv2.imread(p)[:, :, ::-1]
    i = cv2.resize(i, (256, 256), cv2.INTER_LANCZOS4)
    i = np.expand_dims(i, axis=0).astype('float32') / 255.
    return i


ROOT = '/media/bonilla/My Book/115_Paintings/hermitage'
images = glob.glob(os.path.join(ROOT, '*'))
with tqdm.tqdm(total=len(images)) as pbar:
    for image in images:
        try:
            img = load_image(image)
        except Exception as e:
            print(e)
            print(image)
            continue
        maps = model.predict(img)[0, :, :, 0]
        heatmap = np.uint8(maps * 255.)
        cv2.imwrite(f'./paintings_maps/{image.split(os.sep)[-1]}', heatmap)
        pbar.update()
