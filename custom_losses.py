import tensorflow as tf


class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super().__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):

        '''
        Loss for the bounding boxes:
        for the prediction outlier ( > delta) the loss is proportionnal to the distance
        to avoid to correct brutally the outlliers.
        for the loss < delta the loss is square to improve as much as possible the learning
        '''

        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    # For classification if error is small (< self._delta) error must become very small (the sum of all this error must
    # be close to 0 for all the image where the loss is not important, even if there is 100000 * small loss

    def __init__(self, alpha, gamma):
        super().__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma # puissance a laquel est elevee le staus erreur de chaque label entre 0 et 1
        # pour reduire un maximum la grande somme des petites erreurs proches de 0

    def call(self, y_true, y_pred):

        '''
        Loss for the label y_true and y_pred are one hot-vectors
        '''

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits( # loss between onehot y_pred and onehot y_true
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred) # [0 - 1] shape = shape(y_pred)
        # make ponderation of the y_true to accentuate the '1', which become alpha)
        # Useful if there is lots of y_true = 0 and few y_true = 1
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)  # in the best world all elements of pt should be 1 ?
        # if error close to 0 (ex:0,1 -> loss = 0.000...) -> sum af lots of element well defined but nor perfect
        # (0.1 instead of 0) will remain close to 0 even if the loss involves 10 000 000 of values
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        super().__init__(reduction="auto", name="RetinaNetLoss")

        # Use 2 different classes to estimate the loss:
        # 1 class for the label loss
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)

        # 1 class for the bouding box loss
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):


        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4] # photo, anchor, prediction
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot( # Initially, the labels are encoded in y_true with ordinal values in the fifth column.
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32) # all the anchor with a prediction > 1
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32) # all the ignored anchor (pred betwwen (0.4 and 0.5)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss) # for the label : no loss for the label ignored
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0) # for the bounding box: only loss where the labels is defined with p > 0.5
        normalizer = tf.reduce_sum(positive_mask, axis=-1) # sum number of labels existing
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss