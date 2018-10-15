import os

import numpy as np
import keras
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tqdm import tqdm


class TextSequenceGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, mode="train", batch_size=16,
                 img_size=(224, 224), no_channels=3, shuffle=True):
        # train 95, test 5
        self.imgs, self.labels = [], []
        if mode == "train":
            base_train = 'path-to-train-folder/train'
            for folder in tqdm(id_labels):
                label_path = os.path.join(base_train, folder, 'images')
                fn_paths = sorted(os.listdir(label_path))
                for fn_path in fn_paths:
                    self.imgs.append(os.path.join(label_path, fn_path))
                    self.labels.append(folder)
        elif mode == "val":
            base_val = 'path-to-val-folder/val'
            with open('path-to-val-folder/val/val_annotations.txt') as f:
                for line in f:
                    fn_path = os.path.join(
                        base_val, "images", line.split('\t')[0])
                    id_label = line.split('\t')[1]
                    self.imgs.append(fn_path)
                    self.labels.append(id_label)

        self.ids = range(len(self.imgs))

        self.img_size = img_size
        self.img_w, self.img_h = self.img_size
        self.batch_size = batch_size
        self.no_channels = no_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        ids = [self.ids[k] for k in indexes]

        X, y = self.__data_generation(ids)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids):
        size = len(ids)

        X = np.empty(
            (size, self.img_w, self.img_h, self.no_channels),
            dtype=np.float32
        )
        Y = np.empty((size, 100), dtype=np.float32)

        for i, id_ in enumerate(ids):
            img = image.load_img(self.imgs[id_], target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            X[i, ] = img
            Y[i] = wv_label_mapping[self.labels[id_]]
        return X, Y
