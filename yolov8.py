#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# The high rate of traffic accidents in Indonesia, particularly those involving motorcycles, has become a serious issue that requires urgent attention. One of the main contributing factors to the high fatality rate in these accidents is traffic violations, such as riding without wearing a protective helmet.
# 
# This project aims to develop an automated detection system to monitor helmet usage among motorcycle riders on public roads using **deep learning-based object detection methods**. The goal is to support more efficient and systematic traffic surveillance efforts and serve as an initial step toward implementing an **AI-powered automated e-Ticketing (e-Tilang) system**.

# ## Setup

# In[ ]:


import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
from roboflow import Roboflow


# ## Hyperparameters

# In[ ]:


SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0


# ## Load Data

# In[ ]:


class_ids = [
    "motor",
    "helm",
    "non-helm"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Path to images and annotations
path_images = r"./Dataset/helmonzy-5"
path_annot = r"./Dataset/helmonzy-5"

# Get all XML file paths in path_annot and sort them
xml_files = sorted(
    [
        os.path.join(path_annot, file_name)
        for file_name in os.listdir(path_annot)
        if file_name.endswith(".xml")
    ]
)

# Get all JPEG image file paths in path_images and sort them
jpg_files = sorted(
    [
        os.path.join(path_images, file_name)
        for file_name in os.listdir(path_images)
        if file_name.endswith(".jpg")
    ]
)


# In[ ]:


def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(path_images, image_name)

    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return image_path, boxes, class_ids


image_paths = []
bbox = []
classes = []
for xml_file in tqdm(xml_files):
    image_path, boxes, class_ids = parse_annotation(xml_file)
    image_paths.append(image_path)
    bbox.append(boxes)
    classes.append(class_ids)


# In[ ]:


bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))


# Splitting data in training and validation data

# In[ ]:


# Determine the number of validation samples
num_val = int(len(xml_files) * SPLIT_RATIO)

# Split the dataset into train and validation sets
val_data = data.take(num_val)
train_data = data.skip(num_val)


# In[ ]:


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.ragged.map_flat_values(tf.cast, classes, tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


# ## Data Augmentation
# 

# In[ ]:


augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
        keras_cv.layers.RandomShear(
            x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy"
        ),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy"
        ),
    ]
)


# In[ ]:


def ragged_to_dense(inputs):
    boxes = inputs["bounding_boxes"]["boxes"].to_tensor(default_value=0.0)
    classes = inputs["bounding_boxes"]["classes"].to_tensor(default_value=-1.0)

    bounding_boxes = {
        "boxes": boxes,
        "classes": classes
    }

    return inputs["images"], bounding_boxes


# ## Creating Training Dataset

# In[ ]:


train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(ragged_to_dense, num_parallel_calls=tf.data.AUTOTUNE)

# Prefetch
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


# ## Creating Validation Dataset

# In[ ]:


resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(0.75, 1.3),
    bounding_box_format="xyxy",
)

val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(BATCH_SIZE * 4)
val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(ragged_to_dense, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# ## Visualization

# In[ ]:


def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )


visualize_dataset(
    train_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)

visualize_dataset(
    val_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)


# We need to extract the inputs from the preprocessing dictionary and get them ready to be
# fed into the model.

# In[ ]:


#def dict_to_tuple(inputs):
#    return inputs["images"], inputs["bounding_boxes"]


#train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
#train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

#val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
#val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# ## Creating Model

# In[ ]:


backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone_coco"  # We will use yolov8 small backbone with coco weights
)


# In[ ]:


yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=1,
)


# ## Compile the Model

# In[ ]:


optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

yolo.compile(
    optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
)


# ## COCO Metric Callback

# In[ ]:


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)  # Save the model when mAP improves

        return logs


# In[ ]:


for x, y in train_ds.take(1):
    print("Image shape:", x.shape)
    print("Label keys:", y.keys())
    print("Boxes:", y["boxes"])
    print("Classes:", y["classes"])


# In[ ]:


yolo.summary()


# ## Train the Model

# In[ ]:


yolo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    #callbacks=[EvaluateCOCOMetricsCallback(val_ds, "model.h5")],
)


# ## Visualize Predictions

# In[ ]:


def visualize_detections(model, dataset, bounding_box_format):
    # Ambil satu batch dari dataset
    batch = next(iter(dataset.take(1)))

    images, y_true = next(iter(dataset.take(1)))


    # Prediksi menggunakan model
    y_pred = model.predict(images)

    # Langsung plot tanpa ubah ke ragged
    keras_cv.visualization.plot_bounding_box_gallery(
        images=images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        rows=2,
        cols=2,
        scale=4,
        font_scale=0.7,
        class_mapping=class_mapping,
    )


visualize_detections(yolo, dataset=val_ds, bounding_box_format="xyxy")


# In[ ]:


get_ipython().system('pip install pipreqs')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('pipreqs "/content/drive/MyDrive/Colab Notebooks/yolov8/" --scan-notebooks')

