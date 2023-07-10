

import tensorflow as tf
import pandas as pd

class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

The deeper we go in the network, the smaller the dimension of the feature map,
 and the larger the stride between each anchor on the original photo format.


    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):

        # The anchors are generated from 3 different shapes at 3 different scales.
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)

        # The feature maps are generated at layers 3, 4, 5, 6, and 7.
        # self._strides factor of reduction of the image for each level of feature map
        self._strides = [2 ** i for i in range(3, 8)]


        # chaque anchor du meme niveau aura la meme aire, adapte les dimensions en fonction de la forme
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims() # return array  (1, 1, 9, 2) of the dimension of 9 each anchors per level

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            # len(anchor_dims) will be 9 ((3 scales * 3 shapes)
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    # list de tensor dim = (1,1,2) qui represente les dimension des anchor
                    anchor_dims.append(scale * dims)

            # shape anchor_dim_all = (1, 1, nbr_forme_differente, 2)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))

        # return list of the dimension of 9 each anchors per level
        return anchor_dims_all


    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Create tensor of shape (nombre d'anchor possible x hauter du feature map x largeur feature map, 4)
        [[x_center of the anchor, y_center of the anchor, height of the anchor, width of height of the anchor]]
        there is 9 anchor different for each element of the feature map

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        #Grid the center of each feature map by its length and width.
        # feature_width  = 64
        # rx = [0.5, 1.5 ..., 63.5]
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        # Generate a 2D array with the x, y coordinates of the centers.
        # level used : [3, 4, 5, 6, 7] -> to use them in indice need -3
        # position of the center or set to match with the original size of the image
        # dim(centers) = (feature_height, feature_width, 2)
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3] # met le centre des anchors a l'echelle de l'image de base

        # dim(centers) = (feature_height, feature_width, 1,  2)
        centers = tf.expand_dims(centers, axis=-2)

        # dim(centers) = (feature_height, feature_width, 9,  2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])

        # shape(dims) = (feature_height, feature_width, 9,  2) = (1, 1, 9, 2) * (feature_height, feature_width, 1,  1)
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        # shape(anchors) = (feature_height, feature_width, 9,  4)
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )


    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0) # shape (total_anchors, 4)

if __name__ == '__main__':
    A = AnchorBox()
    A._get_anchors(feature_height=32, feature_width=32, level=4)


    # x = tf.constant([1, 2, 3])  # Shape: (3,)
    # y = tf.constant([4, 5, 6])  # Shape: (3,)
    # z = tf.constant([7, 8, 9])  # Shape: (3,)
    #
    # # Empilage des tensors
    # stacked = tf.stack([x, y, z], axis=-1)
    #
    # # Affichage du tensor empilé
    # print("Stacked tensor:")
    # print(stacked.numpy())
    # pd.set_option('display.max_columns', None)
    # df = pd.read_csv('colon.csv')
    # print(df.head(5))