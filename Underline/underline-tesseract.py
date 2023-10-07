import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import json

# Load the YOLO model
from ultralytics import YOLO
model = YOLO('underline.pt')

def process_image(file_path):
    # Run inference using the YOLO model
    results = model(file_path)
    image = cv2.imread(file_path)

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

    cv2.imwrite("image.jpg", canvas)
    # Convert the merged image to grayscale for OCR
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Use pytesseract for OCR on the merged image
    extracted_text = pytesseract.image_to_string(gray_canvas, output_type=Output.STRING)

    return extracted_text

if __name__ == "__main__":
    # Replace 'your_input_image.jpg' with the path to your input image
    input_image_path = 'u1.jpg'
    extracted_text = process_image(input_image_path)
    print(extracted_text)