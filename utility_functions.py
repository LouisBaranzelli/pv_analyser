
import tensorflow as tf


def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    # Calculation of the center, calculation of the width and height
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    exemple:
    tensor = tf.constant([[10, 10, 2, 1]], dtype=tf.float32)
    print(convert_to_corners(tensor))
    >> tf.Tensor([[ 9.   9.5 11.  10.5]], shape=(1, 4), dtype=float32)

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """


    # selecting the intersection of two images.
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    # lu is x1, y1 of the intersection
    # shape(boxes1_corners) = (M x 4), shape(boxes2_corners) = (N x 4)
    # shape(intersection) shape = (M x N x 4) => None allow the broadcasting
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    # rd is x2, y2 of the intersection
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    # shape(rd, lu) = (M x N x 4)
    # Calculate the sides of the intersection
    # If the values are negative, it means that the intersection is null.
    # if negatif -> 0
    intersection = tf.maximum(0.0, rd - lu)
    # intersection_area = M x N x 1 = (M x N x intersection area)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]


    # Calculate the sides of each box
    boxes1_area = boxes1[:, 2] * boxes1[:, 3] # Shape (N,)
    boxes2_area = boxes2[:, 2] * boxes2[:, 3] # Shape (M,)

    # A UNION B = A + B - A INTER B (minimum 1e-8)
    # remark#1 :  Shape (3, 1) +  Shape (3, 1) =  Shape (3, 1)
    # remark#2 :  Shape (3, 1) +  Shape (1, 3) =  Shape (3, 3) -> broadcasting

    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    # Resize betweem 0 and 1, shape = (M x N x 1)
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)