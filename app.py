# Imports Needed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# We resize the images and masks
def resize(input_image, input_mask, size=(128, 128)):
    input_image = tf.image.resize(input_image, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_mask = tf.image.resize(input_mask, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, input_mask


# Augmentation by flipping horizontally
def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    return input_image, input_mask


# Normalization for the images by scaling the images to the range of [-1, 1] and decreasing the image mask by 1
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


# We create two functions to preprocess the training and test datasets with a slight difference between the two
def load_image_train(datapoint):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = augment(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def load_image_test(datapoint):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask
