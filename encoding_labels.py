
import tensorflow as tf
import anchor_box
import utility_functions as uf



class LabelEncoder:
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):

        # Generate different anchor boxes on the original photo for each level,
        # placed regularly on the image with a stride that depends on the level
        # at which the anchor is generated.
        self._anchor_box = anchor_box.AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32 # coefficient pour augmenter les valeur des difference
            # entre les anchor et les anchors true (x_centre, y_centre, height, width)
        )

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during : object is not noise
          but not enough visible to make prediction
            training
        """
        iou_matrix = uf.compute_iou(anchor_boxes, gt_boxes) # shape = (n_anchor_boxes * m_gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1) # for each anchor what is the value best matching (IOU) true box
        matched_gt_idx = tf.argmax(iou_matrix, axis=1) # indice of the corresponding true box for each anchor
        positive_mask = tf.greater_equal(max_iou, match_iou) #  only anchor with matching > 0.5
        negative_mask = tf.less(max_iou, ignore_iou) #  only anchor with matching < 0.4
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask)) # element between (0.4 and 0.5)

        # M elements with:
        # indice of the corresponding true box per anchor,
        # if the anchor contains some thing interesting inside
        # if the anchor NOT contains some thing interesting inside
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        # calculation of:
        # Ratio translation offset in x, Ratio translation offset in y,
        # Logarithm of width ratios, Logarithm of height ratios
        # Ratio translation offset in x : Δ (both center abcise / abcisse of the anchor
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """
        Creates box and classification targets for a single sample
        Return for each anchor (if match: IOU > 0.5) list of the label with performance.
        if no match or weak match (-2 or 12)
        """

        # Create the anchor on the images
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        # matched_gt_idx : list of the indice of the true label
        # positive_mask : anchor which have a match
        # ignore_mask : anchor which have no match

        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )


        # tf.print(matched_gt_idx, summarize=-1)
        # tf of the size of n_anchor in which there is the coordinate of the best true box (AOU)
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)

        # Calculation of matching (per coordonny) between anchor boxes and the best true box associated
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)

        # For each anchor, get the label of the best true box associated
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)

        # tensor with : if anchor match with a true box => label of this true box, if no match => -1)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        # if ignore = -2, if match between (0,4 and 0.5) => -1, else indicate the label
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)

        # Associate the matching result of proximity and the label
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """
        Creates box and classification targets for a batch

        image :  batch_size, height_im, width_image, channel
        bbox : batch_size, anchor , (x1, y1 ,width , height)
        class_id batch_size, label)

        """
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        # create dynamic array
        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):

            # label: for each anchor if match (kIOU > 0.5) list of the label with performance.
            #         if no match or weak match (-2 or 12)
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        # convert pixel in float32, normalize and centralize value of the pixel
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack() # labels.stack align values a longside the axe 0