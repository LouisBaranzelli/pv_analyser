from feature_pyramid import FeaturePyramid
from build_head import build_head
from tensorflow import keras
import tensorflow as tf
import numpy as np


class RetinaNet(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super().__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone) # initialization of backbone and of the layers for the Feature Pyramide Network
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        # Create classification head and the detection head
        self.cls_head = build_head(9 * num_classes, prior_probability) # 9 because 9 anchors
        self.box_head = build_head(9 * 4, "zeros") # 4 : number of element of the bounding box

    def call(self, image, training=False):
        # fpn -> Create the FPN -> initialise the backbone with the size of the image
        features = self.fpn(image, training=training) # Initialize the resnet network with the size of the image ?
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features: # for each output of the FPN

            # shape(box_head) = # l x L  of the image at the level x n_class * nbr_type_anchor
            # box_outputs : per image value for each anchor, the position of the 4 points to define position of the anchor
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4])) # flatten each filter (1 filter = number of element of the bounding box)
            # cls_outputs : per image for each anchor define the label associated
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)  # [N, 9, num_classes]) -> [N, 9*num_classes]
        box_outputs = tf.concat(box_outputs, axis=1)  # [N, 9, 4]) -> [N, 36]

        # for each image :
        # 36 fist column : position of the 9 anchors
        # next 9*num_classes : label per anchor
        return tf.concat([box_outputs, cls_outputs], axis=-1)



if __name__ == '__main__':
   print('Hello')