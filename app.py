import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#@st.cache(allow_output_mutation = True)

model = load_model('inceptionV3.h5')

img_size = (150, 150)
last_conv_layer_name = 'mixed7'

st.title('Malaria-Detection')

def load_image(image_file):
	img = Image.open(image_file)
	return img

def get_img_array(pil_img, img_size):
    # `img` is a PIL image of size 150x150
    img = pil_img.resize(img_size, Image.ANTIALIAS)
    # `array` is a float32 Numpy array of shape (150, 150, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 150, 150, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def gradcam(pil_img, heatmap, alpha=1):
    # Load the original image
    img = keras.preprocessing.image.img_to_array(pil_img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img.resize((250,250), Image.ANTIALIAS)

st.subheader("Image")
image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

try:
    if image_file is not None:
        # To View Uploaded Image
        st.image(load_image(image_file).resize((250,250), Image.ANTIALIAS), caption = 'Uploaded Image')

    pil_img = load_image(image_file)
    img_array = (get_img_array(pil_img, img_size))/255
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    if pred_class == 0:
        statement = "REPORT: The cell is Parasitized"
    else:
        statement = "REPORT: The cell is Uninfected."

    st.subheader(statement)
    if pred_class == 0:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        st.image(gradcam(pil_img, heatmap), caption = 'Infected Area')

except:
    pass

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: visible;}
            footer {visibility: hidden;}
            footer:after {
	            content:'Made by: Anubhab Paul | Nilavo Boral | Suvam Bit'; 
	            visibility: visible;
	            display: block;
	            position: relative;
	            #background-color: #DAF7A6 ;
	            padding: 5px;
	            top: 2px;
                color: #11FF00;
}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)