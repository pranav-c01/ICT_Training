import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import io


# Load environment variables for Azure AI credentials
load_dotenv()
ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
ai_key = os.getenv('AI_SERVICE_KEY')


# Authenticate Azure AI Vision client
cv_client = ImageAnalysisClient(endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key))


# Streamlit page setup
st.title("Azure Vision API Image Analysis")


# Upload image using Streamlit
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


# Define a function to analyze the image
def AnalyzeImage(image_data):
    st.write("Analyzing image...")


    try:
        # Analyze image using Azure Vision API
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE],
        )


        # Display analysis results
        if result.caption:
            st.write(f"**Caption**: '{result.caption.text}' (Confidence: {result.caption.confidence * 100:.2f}%)")


        if result.dense_captions:
            st.write("**Dense Captions**:")
            for caption in result.dense_captions.list:
                st.write(f"Caption: '{caption.text}' (Confidence: {caption.confidence * 100:.2f}%)")


        if result.tags:
            st.write("**Tags**:")
            for tag in result.tags.list:
                st.write(f"Tag: '{tag.name}' (Confidence: {tag.confidence * 100:.2f}%)")


        if result.objects:
            st.write("**Objects in the image**:")
            image = Image.open(io.BytesIO(image_data))
            draw = ImageDraw.Draw(image)
            color = 'cyan'


            for detected_object in result.objects.list:
                st.write(f"Object: {detected_object.tags[0].name} (Confidence: {detected_object.tags[0].confidence * 100:.2f}%)")
                r = detected_object.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)


            # Display image with objects highlighted
            st.image(image, caption="Objects detected", use_column_width=True)


        if result.people:
            st.write("**People detected in the image**:")
            image = Image.open(io.BytesIO(image_data))
            draw = ImageDraw.Draw(image)
            color = 'cyan'


            for detected_people in result.people.list:
                r = detected_people.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)


            # Display image with people highlighted
            st.image(image, caption="People detected", use_column_width=True)


    except HttpResponseError as e:
        st.error(f"Error: {e.reason} - {e.message}")




# Run the analysis when the image is uploaded
if uploaded_image is not None:
    image_data = uploaded_image.read()
    AnalyzeImage(image_data)
