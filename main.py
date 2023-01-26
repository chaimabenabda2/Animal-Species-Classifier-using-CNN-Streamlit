import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PIL import Image
# Packages required for Image Classification
from tensorflow import keras
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


st.set_page_config(page_title="Image classification", layout="wide")



#Predict function
def predict(image1):
    model = VGG16()
    image = load_img(image1, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    return label
#function to load jason animation
def load_lottie(url):
    r=requests.get(url)
    if r.status_code !=200 :
        return None
    return r.json()
animal_img=Image.open("C:\Bureau\cours idsd\projetWeb/animals.png")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("C:\Bureau\cours idsd\projetWeb\style.css")

lottie_coding=load_lottie("https://assets8.lottiefiles.com/packages/lf20_kbfzivr8.json")

with st.container():
    st.subheader("Image Classification using Streamlit")
    left_c0, right_c0 = st.columns(2)
    with left_c0:
        st.title("Animal Recognition with CNN")
    with right_c0:
        st_lottie(lottie_coding, height=200)

with st.container():
    st.write('---')
    left_c1 , right_c1=st.columns(2)
    with left_c1:
        st.write("CNN: Using a pre-trained VGG 16 model for animals recognition ")
        st.write("The VGG 16 is a model that has trained on millions of images from the imagenet database. "
                 "This allows to answer a problem on a specific dataset without starting from zero and on the other hand the training time is also much shorter than a complete model.")
        st.write("In our case, the model could probably succeed very well for animal recognition without specific training and it indicates a percentage of its prediction.")
        st.write("[learn more about CNN VGG16](https://www.kaggle.com/code/stephanedc/tutorial-cnn-partie-3-mod-le-vgg16)")
    with right_c1:
        st.image(animal_img)



with st.container():
    st.write('---')

    #Uploading Image
    uploaded_file = st.file_uploader("Choose an image ", type=["jpg", "jpeg", "png", "gif","jfif"])
    #st.file_uploader function is used to create a file uploader widget that accepts image files of type "jpg", "jpeg", "png", and "gif"
    #The selected file is stored in the variable uploaded_file

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', width=300)
        #If a file is uploaded, it is displayed using the st.image function.
with st.container():
        classify = st.button("classify image")
        if classify:
                st.write("")
                st.write("Classifying...")
                label = predict(uploaded_file)
                st.write('%s (%.2f%%)' % (label[1], label[2]*100))
with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
       <form action="https://formsubmit.co/chaimabenabdallah28@gmail.com" method="POST" >
           <input type="hidden" name="_captcha" value="false">
           <input type="text" name="name" placeholder="Your name" required><br>
           <input type="email" name="email" placeholder="Your email" required><br>
           <textarea name="message" placeholder="Your message here" required></textarea><br>
           <button type="submit">Send</button>
       </form>
       """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()
