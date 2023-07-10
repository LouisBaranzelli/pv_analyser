
from tensorflow import keras
import tensorflow as tf

def get_backbone():
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3] # no specific size for the input images + RGB
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )


if __name__ == '__main__':
    import datetime
    import numpy as np
    # test = get_backbone()
    # print(test.summary())
    # print(
    # .__version__)
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # print(tf.config.list_physical_devices('GPU'))

    # t0 = datetime.datetime.now()
    # tf.config.list_physical_devices('CPU')
    # a = tf.random.normal((10000, 10000), mean=0, stddev=1)
    # inverse_tenseur = tf.linalg.inv(a)
    # t1 = datetime.datetime.now()
    #
    # print(t1 - t0)

    x = -1
    assert x > 0, "x should be greater than 0"

