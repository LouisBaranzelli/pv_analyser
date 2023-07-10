import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_detections(
    image, boxes, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    '''
        Visualize Detections
    Args:
        image: tf[height. width, 3]
        boxes [n_anchor, 4]: x_center, y_center, width, height
    Returns:

    '''

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box in boxes:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
    plt.show()
    return ax


