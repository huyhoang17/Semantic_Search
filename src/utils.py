import random
import os
import logging

from annoy import AnnoyIndex
import numpy as np
from numpy import linalg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_glove_vectors(glove_dir, glove_name='glove.6B.100d.txt'):
    embeddings_index = {}
    with open(os.path.join(glove_dir, glove_name)) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def load_paired_img_wrd(labels_map_cleanup):
    wv_label_mapping = {}
    for key, value in labels_map_cleanup.items():
        vectors = np.array([word_vectors[label] if label in word_vectors
                            else np.zeros(shape=100) for label in value])
        class_vector = np.mean(vectors, axis=0)
        wv_label_mapping[key] = class_vector
    return wv_label_mapping


def generate_features(id_labels, model):
    base_train = 'path-to-train-folder/train'
    for folder in tqdm(id_labels):
        label_path = os.path.join(base_train, folder, 'images')
        fn_paths = sorted(os.listdir(label_path))
        fn_paths = [os.path.join(label_path, fn_path) for fn_path in fn_paths]
        for fn_path in fn_paths:
            img = image.load_img(fn_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            input_ = preprocess_input(img)
            feature = model.predict(input_)

            yield feature, fn_path


def build_image_mapping(id_labels):
    base_train = 'path-to-train-folder/train'
    i = 0
    images_mapping = {}
    for folder in tqdm(id_labels):
        label_path = os.path.join(base_train, folder, 'images')
        fn_paths = sorted(os.listdir(label_path))
        fn_paths = [os.path.join(label_path, fn_path) for fn_path in fn_paths]
        for fn_path in fn_paths:
            images_mapping[i] = fn_path
            i += 1
    return images_mapping


def build_word_mapping(word_vectors):
    word_list = [(i, word) for i, word in enumerate(word_vectors)]
    word_mapping = {k: v for k, v in word_list}
    return word_mapping


def index_features(features, mode="image", n_trees=1000, dims=4096):
    feature_index = AnnoyIndex(dims, metric='angular')
    for i, row in enumerate(features):
        vec = row
        if mode == "image":
            feature_index.add_item(i, vec[0][0])
        elif mode == "word":
            feature_index.add_item(i, vec)
    feature_index.build(n_trees)
    return feature_index


def extract_feat(self, img_path):
    img = image.load_img(img_path, target_size=(
        self.input_shape[0], self.input_shape[1]))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feat = self.model.predict(img)
    norm_feat = feat[0] / linalg.norm(feat[0])
    return norm_feat


def search_index_by_key(key, feature_index, item_mapping, top_n=10):
    distances = feature_index.get_nns_by_item(
        key, top_n, include_distances=True
    )
    return [[a, item_mapping[a], distances[1][i]]
            for i, a in enumerate(distances[0])]


def show_sim_imgs(search_key, feature_index, feature_mapping):

    results = search_index_by_key(
        search_key, feature_index, feature_mapping, 10
    )

    main_img = mpimg.imread(results[0][1])
    plt.imshow(main_img)
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3
    for i in range(1, columns * rows + 1):
        img = mpimg.imread(results[i][1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def show_imgs(id_folder):
    fmt = 'path-to-train-folder/train/{}/images/{}_{}.JPEG'
    random_imgs = [fmt.format(id_folder, id_folder, num)
                   for num in random.sample(range(0, 500), 9)]
    fig = plt.figure(figsize=(16, 16))
    columns = 3
    rows = 3
    for i in range(1, columns * rows + 1):
        img = mpimg.imread(random_imgs[i - 1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def remove_punctuation(text):
    """https://stackoverflow.com/a/37221663"""
    import string  # noqa
    table = str.maketrans({key: None for key in string.punctuation})
    return text.translate(table)
