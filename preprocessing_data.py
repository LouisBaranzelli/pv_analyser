
import tensorflow as tf
import utility_functions as uf

def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.
    50% chance to make horizontal inversion of the image and adap the bbouding box also
    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        # Generates a random value for size between two values to resize and augment the dataset.
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)

    # select a valid ratio to resize the image randomly
    # select also the smallest side of the image and make a ratio
    ratio = min_side / tf.reduce_min(image_shape)
    # If the ratio does not exceed the longest side, redefines the ratio with the maximum value.
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)

    # resize the image
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))

    # Adapt the size of the final image to match with the stride
    # The size of the image is a multiple of the stride of the lowest feature map
    # if the image smaller, add marg with 0
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    # padded_image_shape est la taille de l'image souhaite
    # padded_image_shape[0], padded_image_shape[1] sonnt les dim finale: rembourage de l'image avec des 0 pour atteindre cette taille
    # padded_image_shape
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio


def preprocess_data(sample):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = sample["image"]

    # initially bbox are x1, y1, x2, y2 -> in matrice :(y1, x1, y2, x2)
    bbox = uf.swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    # image, bbox = random_flip_horizontal(image, bbox) # Useless because done in my dataset

    # resize the image
    image, image_shape, ratio = resize_and_pad_image(image)

    # resize the box
    bbox = tf.cast(bbox, tf.float32) * ratio

    bbox = uf.convert_to_xywh(bbox)
    return image, bbox, class_id