import boto3
import google.generativeai as genai
import os
import logging
import time
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)


@app.before_request
def log_request():
    app.logger.debug(f"Request to {request.path}")


#AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
#AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEYR")

AWS_ACCESS_KEY=""
AWS_SECRET_KEY=""
GEMINI_API_KEY=""

try:
    rekognition = boto3.client(
        'rekognition',
        region_name='us-east-1',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    
    logging.info("AWS Rekognition connected succesfully.")
except Exception as e:
    logging.error(f"Error connecting to AWS Rekognition: {e}")
    rekognition = None

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    #model.generate_content("Test connection")
    logging.info("Gemini API connected succesfully.")
except Exception as e:
    logging.error(f"Error connecting to Gemini API: {e}")
    genai = None

def detect_food_labels(image_bytes):
    response = rekognition.detect_labels(
        Image={'Bytes': image_bytes},
        MaxLabels=10,
        MinConfidence=70
    )
    labels = [label['Name'] for label in response['Labels']]

    with open("detected_food.log", "a") as log_file:
        log_file.write("Detected food items: " + ", ".join(labels) + "\n")

    return labels


def generate_recipe(ingredients):
    if not genai:
        return "Error: connection did not establish well."

    prompt = f"Create a simple recipe using: {', '.join(ingredients)}."
    max_retries = 5
    delay = 2

    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Error generating recipe: {e}")
                return "Error genereting recipe."


from flask import render_template


@app.route("/")
def index():
    return render_template("index.html") 


@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    image = request.files["file"].read()

    food_items = detect_food_labels(image)
    if not food_items:
        return jsonify({"error": "No food items detected."}), 400

    try:
        recipe = generate_recipe(food_items)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"ingredients": food_items, "recipe": recipe})

PORT = int(os.getenv("PORT", 8080))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
