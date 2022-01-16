from datetime import datetime
from inspect import trace
import os

import tensorflow as tf
import tensorflow_datasets as tfds

print("Tensorflow Version is ", tf.__version__)

def getDataset():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    def normalize_img(image, label):
        """ Normalizes images : uint8 > float32"""
        return tf.cast(image, tf.float32)/255, label
    ds_train = ds_train.map(normalize_img).shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)

    ds_test = ds_test.map(normalize_img).batch(6)
    return (ds_train, ds_test)

def getModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28, 1]),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')]
        )
    return model

def main():
    (ds_train, ds_test) = getDataset()
    model = getModel()
    # '编译'模型
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    logs = "logs/"+datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logs, profile_batch='20, 40')
    
    model.fit(ds_train, epochs=5, validation_data=ds_test, callbacks=[tb_callback])

if __name__ == '__main__':
    main()






