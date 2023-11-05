import streamlit as st
import requests
import base64
import json
from PIL import Image, ImageDraw
import io

# Function to convert image to base64
def img_to_base64_str(img: Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
# Function to convert base64 to PIL Image
def base64_str_to_img(base64_str: str) -> Image:
    decoded = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(decoded))
    return img

# Streamlit interface
st.sidebar.title("Object Detection")

# Endpoint URL
endpoint_url = st.sidebar.text_input("Endpoint URL", value="http://0.0.0.0:8000/detect")

# File uploader allows user to add their own image
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Text input for text queries
text_queries = st.sidebar.text_area("Enter text queries, separated by commas")

# Button to send the request
if st.sidebar.button("Detect Objects"):
    if uploaded_file is not None and text_queries:
        # Convert the file to an image
        image = Image.open(uploaded_file)

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Prepare the texts as a list of lists
        texts = [text.strip().split(",") for text in text_queries.split("\n") if text.strip() != ""]

        # Prepare the request payload
        payload = {
            "image_data": image_data,
            "texts": texts
        }

        response = requests.post(endpoint_url, json=payload)

        # Print the response (for debugging purposes)
        st.write("API Response:")
        

        if response.status_code == 200:
            # Get the base64 encoded image with boxes from the response
            img_with_boxes_str = response.json()["image_with_boxes"]
            image_with_boxes = base64_str_to_img(img_with_boxes_str)

            # Display the detections
            detections = response.json()["detections"]
            for detection in detections:
                st.write(detection)

            # Display image with boxes
            st.image(image_with_boxes, use_column_width=True)

        else:
            st.error(f"Failed to detect objects: {response.status_code}")
            st.json(response.json())
    else:
        st.error("Please upload an image and enter some text queries.")