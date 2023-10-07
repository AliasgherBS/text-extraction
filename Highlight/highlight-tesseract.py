import cv2
import numpy as np
import json
import boto3

# Load the YOLO model
from ultralytics import YOLO
model = YOLO('highlight.pt')

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

    # Convert the image to bytes
    _, img_encoded = cv2.imencode('.jpg', gray_canvas)
    image_bytes = img_encoded.tobytes()

    # Perform OCR on the merged image

    # Add API & Secret key here
    client = boto3.client('textract', aws_access_key_id="your api key", aws_secret_access_key='your secret code', region_name="us-west-2")    
    response = client.detect_document_text(Document={'Bytes': image_bytes})

    extracted_text = ''

    # Extract and concatenate the detected text
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            extracted_text += item['Text']

    return extracted_text

if __name__ == "__main__":
    # Replace 'your_input_image.jpg' with the path to your input image
    input_image_path = 'h1.jpg'
    extracted_text = process_image(input_image_path)
    print(extracted_text)