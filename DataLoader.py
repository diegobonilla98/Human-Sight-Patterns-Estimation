from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imgaug as ia
import imgaug.augmenters as iaa


class DataLoader(Sequence):
    def __init__(self, batch_size, size):
        self.batch_size = batch_size
        self.size = size
        self.ROOT = '/media/bonilla/My Book/WherePeopleLook'
        self.MAPS = glob.glob(os.path.join(self.ROOT, 'ALLFIXATIONMAPS', '*'))
        self.MAPS = sorted(list(filter(lambda x: x.endswith('_fixMap.jpg'), self.MAPS)))
        self.STIMULI = sorted(glob.glob(os.path.join(self.ROOT, 'ALLSTIMULI', '*')))

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-45, 45),
                    shear=(-16, 16),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),
                iaa.SomeOf((0, 5),
                           [
                               sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),
                                   iaa.AverageBlur(k=(2, 7)),
                                   iaa.MedianBlur(k=(3, 11)),
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                               iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True),
                               iaa.Add((-10, 10), per_channel=0.5),
                               iaa.AddToHueAndSaturation((-20, 20)),
                               iaa.OneOf([
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                   iaa.FrequencyNoiseAlpha(
                                       exponent=(-4, 0),
                                       first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                       second=iaa.LinearContrast((0.5, 2.0))
                                   )
                               ]),
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

    def __len__(self):
        return len(self.MAPS) // self.batch_size

    def __getitem__(self, item):
        X = []
        Y = []
        idxs = np.random.randint(0, len(self.MAPS), self.batch_size)
        for i in idxs:
            stimuli = self.STIMULI[i]
            stimuli = cv2.imread(stimuli)[:, :, ::-1]
            stimuli = cv2.resize(stimuli, self.size)
            maps = self.MAPS[i]
            maps = cv2.imread(maps, 0)
            maps = cv2.resize(maps, self.size).astype('float32') / 255.
            stimuli, maps = self.seq(image=stimuli, heatmaps=maps[np.newaxis, :, :, np.newaxis])
            stimuli = stimuli.astype('float32') / 255.
            maps = maps[0]
            X.append(stimuli)
            Y.append(maps)
        return np.array(X, np.float32), np.array(Y, np.float32)


if __name__ == '__main__':
    dl = DataLoader(4, (512, 512))
    dx, dy = dl[0]

    plt.figure(0)
    plt.imshow(np.vstack([dx[0], dx[1], dx[2]]))
    plt.figure(1)
    plt.imshow(np.vstack([dy[0, :, :, 0], dy[1, :, :, 0], dy[2, :, :, 0]]))
    plt.show()
