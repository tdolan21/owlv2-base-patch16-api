from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
import random
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import logging
from fastapi.middleware.cors import CORSMiddleware
from logging.handlers import RotatingFileHandler
import base64
import io
import os

# Set up logging with rotating file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    handlers=[
        RotatingFileHandler('logs/testing.log', maxBytes=10000, backupCount=5),
        logging.StreamHandler()
    ]
)

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

class DetectionRequest(BaseModel):
    image_data: str
    texts: List[List[str]]

class DetectionResult(BaseModel):
    detections: List[str]
    image_with_boxes: str

app = FastAPI()

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def draw_bounding_boxes(image: Image, boxes, scores, labels, text_labels):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Define the color bank
    color_bank = ["#0AC2FF", "#47FF0A", "#FF0AC2", "#ADD8E6", "#FF0A47", "#C2FF0A", "#87CEFA", "#778899", "#6A5ACD", "#FF69B4"]

    # Use default font
    font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        # Choose a random color
        color = random.choice(color_bank)

        # Convert the box to a Python list if it's not already
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        elif not isinstance(box, (list, tuple)):
            raise TypeError("Box must be a list or tuple of coordinates.")

        # Draw the rectangle
        draw.rectangle(box, outline=color, width=2)

        # Get the text to display
        display_text = f"{text_labels[label]}: {score:.2f}"

        # Calculate position for the text
        text_position = (box[0], box[1] - 10)

        # Draw the text
        draw.text(text_position, display_text, fill=color, font=font)

    return image



from fastapi.responses import JSONResponse

@app.post("/detect", response_model=DetectionResult)
async def detect_objects(request: DetectionRequest):
    try:
        # Decode the base64 image
        image_data_bytes = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(image_data_bytes))
        width, height = image.size

        inputs = processor(text=request.texts, images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

        detection_strings = []
        image_with_boxes = image.copy()  # Copy the image only once

        for i, text_group in enumerate(request.texts):
            results_per_group = results[i]
            # Verify the model outputs normalized coordinates; otherwise, adjust the boxes here
            boxes = results_per_group["boxes"]
            scores = results_per_group["scores"]
            labels = results_per_group["labels"]

            # Draw bounding boxes and labels on the image without additional scaling
            image_with_boxes = draw_bounding_boxes(image_with_boxes, boxes, scores, labels, text_group)

        # Generate detection strings for each detection
        for box, score, label in zip(boxes, scores, labels):
            scaled_box = [round(box[i].item() * (width if i % 2 == 0 else height), 2) for i in range(len(box))]
            detection_string = f"Detected {text_group[label]} with confidence {round(score.item(), 3)} at location {scaled_box}"
            detection_strings.append(detection_string)

            # Log to see if boxes are drawn
        logging.info("Bounding boxes and labels have been drawn on the image.")

        # Save the image with bounding boxes and encode it to base64
        buffered = io.BytesIO()
        image_with_boxes.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Log the base64 string length to check if it changed
        logging.info(f"Length of base64 string: {len(img_str)}")

        return DetectionResult(
            detections=detection_strings,
            image_with_boxes=img_str
        )

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error", "details": str(e)}
        )
