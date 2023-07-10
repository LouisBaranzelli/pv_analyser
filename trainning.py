import tensorflow_datasets as tfds
from encoding_labels import LabelEncoder
import tensorflow as tf
from backbone import get_backbone
from custom_losses import RetinaNetLoss
from retinanet import RetinaNet
from preprocessing_data import preprocess_data
import os
import pandas as pd
import copy



def read_image(image_file, id_image):

    tf.print(id_image)

    image_file = str(image_file)

    image = tf.io.read_file(os.path.join(dir_image, image_file))
    image = tf.image.decode_jpeg(image, channels=3)

    return {'image': image,
            'image/filename': image_file,
            'objects': { 'label': label,
                         'bbox': bbox,
                         'id': bbox_id
            }}






model_dir = "retinanet/"

# set list of the dimension of 9 each anchors per level
label_encoder = LabelEncoder()

num_classes = 80
batch_size = 2

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

# Builds ResNet50 with pre-trained imagenet weights
resnet50_backbone = get_backbone()

# initialize both class of loss
loss_fn = RetinaNetLoss(num_classes)

# Add the FPN and head detection to the backbone
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

# CReation of the callback
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
]

#  set `data_dir=None` to load the complete dataset
# train_dataset : PrefetchDataset including
(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)







dir_image = os.path.join(os.getcwd(), 'dataset_pv')
label = []
bbox_id = []
bbox = []

df_csv = pd.read_csv(os.path.join('dataset_pv', 'dataset.csv'))
image_file = df_csv['image'].to_list()
list_id_image = df_csv['id_image'].to_list()
[label.append(eval(each_label)) for each_label in df_csv['label'].values]
[bbox_id.append(eval(each_bbox_id)) for each_bbox_id in df_csv['bbox_id'].values]
[bbox.append(eval(each_bbox_id)) for each_bbox_id in df_csv['bbox'].values]


ds_train = tf.data.Dataset.from_tensor_slices((image_file,
                                               list_id_image))

train_dataset = ds_train.map(read_image)











'''
Creation of the pipeline of data:
1 - Apply the preprocessing function to the samples

2 - Create batches with fixed batch size. Since images in the batch can have different dimensions, 
and can also have different number of objects, we use padded_batch to the add the necessary padding 
to create rectangular tensors

3 - Create targets for each sample in the batch using LabelEncoder
'''

autotune = tf.data.AUTOTUNE  # Dynamically adjust the performance of the data pipeline based on the computer's characteristics and available resources

# Resized and random  flipping applied.
# return image, bbox, class_id : tuple (batch_size, height_im, width_image, channel), (batch_size, anchor , (x1, y1 ,x2 , y2)), (batch_size, label)

# for element in train_dataset:
#     print(element["objects"]["bbox"])

train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)

# shuffle with a window of  8 * batch_size
train_dataset = train_dataset.shuffle(8 * batch_size)

# Cree des btach d'images, drop_remainder=True : Si nombre d'image % batch != 0, ne cree pas un lot plus petit a la fin
# donc tout les lots ont la meme taille
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    # for padding value on rempli le vide
    # des images par 0
    # , bbox par 1e-8
    # class_id par -1
)

# arguments:  image (Tensor de l'images), bbox tensor(nbr_box, 4), class_id tensor(nbr_box,)
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
# return:  tensor(batch_size, image), bbox tensor(batch_size, nbr_anchor, 5 (erreur des 4 coordonnees + label))

train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(
    autotune)  # This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

# if __name__ == '__main__':
#     test = get_backbone()
#     print(test.summary())

# Uncomment the following lines, when training on full dataset
# train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
# val_steps_per_epoch = \
#     dataset_info.splits["validation"].num_examples // batch_size

# train_steps = 4 * 100000
# epochs = train_steps // train_steps_per_epoch

epochs = 1

# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset

model.fit(
    train_dataset.take(100),
    validation_data=val_dataset.take(50),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)

    #





# if __name__ == '__main__':
#     (train_dataset, val_dataset), dataset_info = tfds.load(
#         "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
#     )
#
#
#
#     pass
#


