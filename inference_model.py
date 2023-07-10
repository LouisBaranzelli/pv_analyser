
import tensorflow as tf
from retinanet import RetinaNet
from backbone import get_backbone
from custom_losses import RetinaNetLoss
from preprocessing_data import resize_and_pad_image
import tensorflow_datasets as tfds
from decode_prediction import DecodePredictions
from visualization import visualize_detections
import cv2
import os

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)


num_classes = 80
resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)





# Change this to `model_dir` when not using the downloaded weights
weights_dir = "data"

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)


image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)


detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)


def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
int2str = dataset_info.features["objects"]["label"].int2str


# # Read images in the folder 'dossier_photo'
dict_images = dict()
dossier = 'dossier_photo'
for each_image in os.listdir(dossier):
    path_to_image = os.path.join(os.getcwd(), dossier, each_image)
    # Read image and change color BGR -> RGB
    image = cv2.cvtColor(cv2.imread(os.path.abspath(path_to_image), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
    dict_images[each_image] = tf.cast(image, tf.float32)


for image in dict_images.values():
    # image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )

# From keras
#---------------------------------------------------------
# for sample in val_dataset.take(2):
#     image = tf.cast(sample["image"], dtype=tf.float32)
#     input_image, ratio = prepare_image(image)
#     detections = inference_model.predict(input_image)
#     num_detections = detections.valid_detections[0]
#     class_names = [
#         int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
#     ]
#     visualize_detections(
#         image,
#         detections.nmsed_boxes[0][:num_detections] / ratio,
#         class_names,
#         detections.nmsed_scores[0][:num_detections],
#     )
#---------------------------------------------------------