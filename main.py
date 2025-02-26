import streamlit as st 
import google.generativeai as genai
import os
import PIL.Image

#set api
os.environ['GOOGLE_API_KEY'] = "Your API Key"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

#load model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

#custom function
def analyze_human_attributes(image):
    prompt = """ 
    You are an AI trained to analyze human attributes from Images.So carefully analyze the given images and returned the following details:
    - **Gender** (Male/Female/Non-binary)
    - **Age Estimation** (e.g., 12 years)
    - **Mood** (e.g., Happy,Sad,Neutral,Excited)
    - **Facial Expression** (e.g., Smiling,Frowning,Neutral etc.,)
    - **Glasses** (yes/No)
    - **Beard** (yes/no)
    - **Hair colour** (e.g., black,white,brown)
    - **Eye colour** (e.g., blue,green,brown,black)
    - **Head wear** (yes/no specify type)
    - **Emotions detected** (e.g., joyful,focused,angry etc.,)
    - **Confidence level** (Accuracy of prediction in percentage)
    """
    result = model.generate_content([prompt,image])
    
    return result.text.strip()

#app creation
st.title("Human Attribute Recognition")
st.write("Upload an image to recognize")

#Image upload
uploaded_image = st.file_uploader("Upload an Image", type=['png','jpg','jpeg'])

if uploaded_image:
    img = PIL.Image.open(uploaded_image)
    info = analyze_human_attributes(img)
    
    #st.write(person_info)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption='Uploaded Image', use_container_width=True)
    with col2:
        st.write(info)
