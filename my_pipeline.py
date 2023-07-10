import numpy as np
from utility_functions import convert_to_corners
import tensorflow_datasets as tfds
from encoding_labels import LabelEncoder
import tensorflow as tf
from backbone import get_backbone
from custom_losses import RetinaNetLoss
from retinanet import RetinaNet
from preprocessing_data import preprocess_data
import os
import pandas as pd
import copy, random
import matplotlib.pyplot as plt
from main_image_anchor_vizualisation import visualize_detections

def create_dict_image(dir_csv):
    dir_image = dir_csv
    label = []
    bbox_id = []
    bbox = []

    df_csv = pd.read_csv(os.path.join('dataset_pv', 'dataset.csv'))
    image_file = df_csv['image'].to_list()
    [label.append(eval(each_label)) for each_label in df_csv['label'].values]
    [bbox_id.append(eval(each_bbox_id)) for each_bbox_id in df_csv['bbox_id'].values]
    [bbox.append(tf.stack(eval(each_bbox_id))) for each_bbox_id in df_csv['bbox'].values]

    dataset = {}
    for each_name_image, each_bbox_id, each_bbox, each_label in zip(image_file, bbox_id, bbox, label):
        image = tf.io.read_file(os.path.join(dir_image, str(each_name_image)))
        image = tf.image.decode_jpeg(image, channels=3)
        dict_image = {
            'image': image,
            'image/filename': each_name_image,
            'objects': {'label': each_label,
                        'bbox': each_bbox,
                        'id': each_bbox_id
                        }
        }
        dataset[len(dataset)] = dict_image
    return dataset

def padded_batch_dataset(dic_batch_images: dict):
    '''

    # normalize all the element of the same batch suplied in the dict to get the same dimension

    dict de m_batch_element. Each element is:
        Args:
            image: -> [height, width, canal]
            anchor: -> [number_anchor], [x1, y1, height, width]
            label: (n,) -> [label]

        Returns:
            image: -> [n_im_batch, height, width, canal]
            anchor: -> [number_anchor], [n_im_batch, x1, y1, height, width]
            label:  -> [n_im_batch, label]

    '''

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    for batch_images in dic_batch_images.values():
        max_height, max_width, max_anchor = 0, 0, 0

        # get the max size of the different elements composing this batch
        for each_image_dict in batch_images.values():
            height, width, _ = each_image_dict['image'].shape
            anchor = len(each_image_dict['objects']['label'])
            max_width = max(width, max_width)
            max_height = max(height, max_height)
            max_anchor = max(anchor, max_anchor)

        # pad the different element of the batch to normalize the batch
        for each_image_dict in batch_images.values():
            # pad the image
            each_image_dict['image'] = tf.image.pad_to_bounding_box(
                each_image_dict['image'], 0, 0, max_height, max_width
            )
            # pad bbox with 1e-8
            padding = max_anchor - len(each_image_dict['objects']['label']) # max_anchor - number of anchors for this image
            paddings = tf.constant([[0, padding], [0, 0]])
            each_image_dict['objects']['bbox'] = tf.cast(each_image_dict['objects']['bbox'], tf.float32) # in float32 to be able to add padding with float
            each_image_dict['objects']['bbox'] = tf.pad(
                each_image_dict['objects']['bbox'], paddings,
                mode='CONSTANT', constant_values=1e-8
            )

            # pad label with -1 (label indetermine)
            padding = max_anchor - len(each_image_dict['objects']['label'])
            paddings = tf.constant([[0, padding], [0, 0]])
            each_image_dict['objects']['label'] = tf.reshape(each_image_dict['objects']['label'], (-1, 1)) # dim -> (n, 1)
            each_image_dict['objects']['label'] = tf.reshape(tf.pad(
                each_image_dict['objects']['label'], paddings,
                mode='CONSTANT', constant_values= -1), (-1,))


            # plt.imshow(each_image_dict['image'])
            # plt.show()
    return dic_batch_images


def create_batch(dataset, size_batch):

    '''
    Create random images batch dictionnary of image.
    If number of image % size batch != 0 create 1 smaller batch
    Args:
        dataset: dictionnary of images
        size_batch: number of image / batch

    Returns:

    '''

    dataset_batched = {} # dict of batch of images

    while len(dataset) != 0:
        batch_dict = {} # dict is the batch of images
        for i in range(size_batch):
            if len(dataset) != 0: # the size of the last batch may not be equal to batchsize
                rand_key = random.choice(list(dataset.keys()))
                batch_dict[rand_key] = dataset[rand_key]
                del dataset[rand_key]

        dataset_batched[len(dataset_batched)] = batch_dict
    return dataset_batched

def get_preprocess_data(dict_image):

    '''
    Clean image
    Returns: dict_image cleanned
    '''

    for each_image in dict_image.values():
        each_image['image'], each_image['bbox'], each_image['label'] = preprocess_data(each_image)
        return dict_image

def get_encode_label(dict_batch_image):

    '''
    Merge label with bbox
    Args:
        dict_batch_image: Dict of batches of dict images
    Returns:
        dict_batch_image with keys :
        x: tensor of size batch, height, width, 3
        y: tensor of size batch, height, width, 3
    '''

    x_out, y_out = [], []
    for each_batch in dict_batch_image.values():
        batch_image = []
        batch_label = []
        batch_bbox = []

        for each_image in each_batch.values():
            batch_image.append(each_image['image'])
            batch_label.append(each_image['objects']['label'])
            batch_bbox.append(each_image['objects']['bbox'])

        batch_image = np.stack(batch_image)
        batch_label = np.stack(batch_label)
        batch_bbox = np.stack(batch_bbox)

        batch_image, batch_label = label_encoder.encode_batch(batch_image, batch_bbox, batch_label)
        x_out.append(batch_image), y_out.append(batch_label)

    return x_out, y_out



if __name__ == '__main__':
    # dataset = create_dict_image()

    label_encoder = LabelEncoder()

    dataset = create_dict_image(os.path.join(os.getcwd(), 'dataset_pv'))
    dataset = get_preprocess_data(dataset)
    # visualize_detections(dataset[2]['image'], dataset[2]['objects']['bbox'])
    dataset = create_batch(dataset, 3)
    dataset = padded_batch_dataset(dataset)
    x, y = get_encode_label(dataset)

    a = y[0][0, :, 4]
    indices_lignes = tf.reshape(tf.where(tf.equal(y[0][0, :, 4], 1)),(-1,))
    anchors = tf.gather(y[0], indices_lignes, axis=1)
    anchors = convert_to_corners(anchors[0,: , :4])
    visualize_detections(x[0][0, ...], anchors[:, :4])



    pass
