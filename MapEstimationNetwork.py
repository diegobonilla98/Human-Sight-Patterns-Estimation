import matplotlib.pyplot as plt
from BoniDL import utils, losses
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from DataLoader import DataLoader
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
import CoolUnet
import tensorflow as tf


utils.allow_gpu_growth()


def plot_results(epoch, logs):
    sx, sy = train_generator[0]
    py = model.predict(sx)
    rows, columns = 2, 3
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(sx[0])
    plt.axis("off")
    plt.title("Input1")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(sy[0, :, :, 0])
    plt.axis("off")
    plt.title("GT1")
    fig.add_subplot(rows, columns, 3)
    plt.imshow(py[0, :, :, 0])
    plt.axis("off")
    plt.title("Prediction1")
    fig.add_subplot(rows, columns, 4)
    plt.imshow(sx[1])
    plt.axis("off")
    plt.title("Input2")
    fig.add_subplot(rows, columns, 5)
    plt.imshow(sy[1, :, :, 0])
    plt.axis("off")
    plt.title("GT2")
    fig.add_subplot(rows, columns, 6)
    plt.imshow(py[1, :, :, 0])
    plt.axis("off")
    plt.title("Prediction2")
    plt.savefig(f'./RESULTS/epoch_{epoch}.png')
    plt.close(fig)
    plt.clf()


model = CoolUnet.custom_unet(input_shape=(256, 256, 3), use_attention=False, filters=64, upsample_mode='simple', kernel_initializer=tf.initializers.random_normal(0., 0.02), output_activation='linear')
opt = Adam(lr=5e-4)
model.compile(optimizer=opt, loss=losses.image_euclidean_loss)
model.summary()
plot_model(model, show_shapes=True)

train_generator = DataLoader(8, (256, 256))
callbacks = [LambdaCallback(on_epoch_end=plot_results),
             EarlyStopping(monitor='loss', min_delta=0.001, patience=7, verbose=1, restore_best_weights=True),
             ReduceLROnPlateau(monitor='loss', min_delta=0.002, patience=3, verbose=1)]
model.fit_generator(generator=train_generator, epochs=1000, callbacks=callbacks)
model.save('model.h5', include_optimizer=False)
