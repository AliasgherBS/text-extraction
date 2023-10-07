from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
import json
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from pytesseract import Output

app = FastAPI()

# Enable CORS for all origins (not recommended for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
model = YOLO('highlight.pt')

@app.post("/highlight")
async def process_image(file: UploadFile):
    # Save the uploaded image
    with open("uploaded_image.jpg", "wb") as image_file:
        image_file.write(file.file.read())

    # Run inference using the YOLO model
    results = model("uploaded_image.jpg")
    image = cv2.imread("uploaded_image.jpg")

    # Convert the results to JSON format
    results = results[0].tojson()
    result = json.loads(results)

    # Create an empty canvas to merge cropped images
    canvas = np.zeros_like(image)

    for r in result:
        x1, y1, x2, y2 = (
            int(r["box"]["x1"]),
            int(r["box"]["y1"]),
            int(r["box"]["x2"]),
            int(r["box"]["y2"]),
        )

        # Crop the bounding box area from the original image
        cropped_image = image[y1:y2, x1:x2]

        # Paste the cropped image onto the canvas
        canvas[y1:y2, x1:x2] = cropped_image

    # Convert the merged image to grayscale for OCR
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Use pytesseract for OCR on the merged image
    extracted_text = pytesseract.image_to_string(gray_canvas, output_type=Output.STRING)

    return extracted_text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)