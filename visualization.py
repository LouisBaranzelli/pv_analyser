import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt


def download_dataset():

    url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
    filename = os.path.join(os.getcwd(), "data.zip")
    keras.utils.get_file(filename, url)

    with zipfile.ZipFile("data.zip", "r") as z_fp:
        z_fp.extractall("./")

def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    """Visualize Detections"""

    '''
    boxes = tf.constant([[500, 500, 400, 400]])
    classes = ['test_name']
    scores = [0.5]
    image = cv2.imread('test.png')
    visualize_detections(image, boxes=boxes, classes=classes, scores=scores)

    '''

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax



if __name__ == '__main__':

    boxes = tf.constant([[500, 500, 400, 400]])
    classes = ['test_name']
    scores = [0.5]
    image = cv2.imread('test.png')
    visualize_detections(image, boxes=boxes, classes=classes, scores=scores)
