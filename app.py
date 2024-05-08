import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st

st.header("Brain Tumor Classification Model")
model = load_model('D:\Projects\brain_tumor_classification\Brain_Tumor_Classifier.keras')

data_categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_width = 180
image_height = 180

image = 'glioma.jpg'

img_load = tf.keras.utils.load_img(image, target_size = (image_width, image_height))
img_arr = tf.keras.utils.array_to_img(img_load)
img_batch = tf.expand_dims(img_arr, 0)

predictions = model.predict(img_batch)

score = tf.nn.softmax(predictions)

st.image(image, image_width = 200)
st.write("Brain Tuor in image is " + data_categories[np.argmax(score)])
st.write("The Model Accuracy is" + str(np.max(score)))
