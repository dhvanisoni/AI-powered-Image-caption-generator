"""
IMPORTING LIBRARIES
"""
import re
import ast
from collections import Counter
import base64
import os
import sqlite3
from flask import Flask, render_template, request, redirect

# Access OpenAI API for text generation
import openai
import requests

# Image processing
from ultralytics import YOLO
import cv2
from sklearn.cluster import KMeans
import webcolors
from scipy.spatial import KDTree
import spacy

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["DATABASE"] = "database.db"
app.config["SECRET_KEY"] = "boost is the sectret of my energy"

def create_table():
    """
    Create a 'data' table in the database if it doesn't exist.

    This function connects to the database, executes a SQL command to create the 'data' table
    with appropriate columns (id, text, image), and then commits the changes.

    Returns:
        None
    """
    conn = sqlite3.connect(app.config["DATABASE"])
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS data
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      text TEXT NOT NULL,
                      image BLOB NOT NULL)"""
    )
    conn.commit()
    conn.close()

def create_review_table():
    """
    Create a 'reviews' table in the database if it doesn't exist.

    This function connects to the database, executes a SQL command to create the 'reviews' table
    with appropriate columns (id, rating, comment), and then commits the changes.
    Returns:
        None
    """
    conn = sqlite3.connect(app.config["DATABASE"])
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS reviews
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      rating INTEGER NOT NULL,
                      comment TEXT)"""
    )
    conn.commit()
    conn.close()

def save_image(image):
    """
    Read and return the binary data of an image.

    This function takes an uploaded image file, reads its binary data, and returns it.

    Args:
        image: The uploaded image file.
    Returns:
        bytes: The binary data of the image.
    """
    image_data = image.read()
    return image_data

def load_model(model_path):
    """
    Load a YOLO model from the specified path.

    Args:
        model_path (str): Path to the YOLO model file.

    Returns:
        YOLO: Loaded YOLO model.
    """
    model_yolo = YOLO(model_path)
    return model_yolo

def load_image(image_path):
    """
    Load an image from the specified path using OpenCV.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Loaded image as a NumPy array with RGB color channels.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def detect_objects(model, img, confidence_threshold=0.5):
    """
    Perform object detection on the image using the YOLO model.

    Args:
        model: YOLO model for object detection.
        img: Input image for object detection.
        confidence_threshold (float): Confidence threshold for object detection.

    Returns:
        list: List of detected objects, each represented as a tuple (class_label, confidence_score).
    """
    results = model.predict(img, save=False, save_txt=False)
    no_objects = len(results[0].boxes)
    objects = []
    for i in range(no_objects):
        class_label = results[0].names[results[0].boxes[i].cls[0].item()]
        confidence_score = results[0].boxes[i].conf.item()
        if confidence_score >= confidence_threshold:
            objects.append((class_label, confidence_score))
    return objects

def count_objects(objects):
    """
    Count the number of occurrences for each object class in the list of objects.

    Args:
        objects (list): List of object tuples, where each tuple contains (class, confidence).

    Returns:
        str: A string representing the counts of different object classes.
    """
    counter_result = Counter([obj[0] for obj in objects])
    result_string = str(counter_result)
    result_content = result_string[
        result_string.index("{") + 1 : result_string.index("}")
    ]
    return result_content

def print_objects(objects):
    """
    Print the class label and confidence score for each detected object.

    Args:
        objects (list): List of tuples containing object class and confidence score.

    Returns:
        None
    """
    for obj in objects:
        print(f"\n Class: {obj[0]}, Confidence: {obj[1]}")

def handle_no_objects(image_path):
    """
    Handle the case when no objects are detected in the image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        None
    """
    image = load_image(image_path)
    dominant_color = detect_majority_color(image)
    if dominant_color is not None:
        print("\n Dominant Color:", dominant_color)
        color_name = convert_rgb_to_names(dominant_color)
        print(color_name)
    else:
        print("\n No objects and no dominant color detected.")

def detect_majority_color(image_path):
    """
    Detect the dominant color in an image using k-means clustering.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: RGB values of the detected dominant color.
    """
    image = load_image(image_path)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    color = kmeans.cluster_centers_[0]
    color = color.astype(int)
    return color

def convert_rgb_to_names(rgb_tuple):
    """
    Convert an RGB tuple to the closest matching color name.

    Args:
        rgb_tuple (tuple): RGB values as a tuple (R, G, B).

    Returns:
        str: Closest matching color name.
    """
    colors = webcolors.CSS3_NAMES_TO_HEX
    rgb_values = [webcolors.hex_to_rgb(hex_value) for hex_value in colors.values()]
    color_names = list(colors.keys())
    kdtree = KDTree(rgb_values)
    distance, index = kdtree.query(rgb_tuple)
    closest_color_name = color_names[index]
    return closest_color_name

def process_input(user_input):
    """
    Process user input to handle different types (string, dictionary, list).

    Args:
        user_input (str, dict, list): The user input to be processed.

    Returns:
        str: The processed input as a single string.
    """
    if isinstance(user_input, str):
        variable = user_input
        return variable
    elif isinstance(user_input, dict):
        variables = []
        for key, value in user_input.items():
            variables.append(str(value) + " " + key)

        if len(variables) == 1:
            variable = variables[0]
            return variable
        else:
            variable = ", ".join(variables[:-1]) + " and " + variables[-1]
            return variable

    elif isinstance(user_input, list):
        if all(isinstance(item, str) for item in user_input):
            variable = process_input(user_input[0])
            return variable
    else:
        print("Invalid input! Please enter a word or a dictionary of variables.")

def check_tone(user_tone):
    """
    Check if the user-specified tone is valid.

    Args:
        user_tone (str): The tone specified by the user.

    Returns:
        str or int: The valid user-specified tone if it is in the list of valid tones, otherwise 0.
    """
    tones = [
        "Playful",
        "Adventurous",
        "Inspiring",
        "Serene",
        "Witty",
        "Romantic",
        "Motivational",
        "Humble",
        "Mysterious",
        "Energetic",
        "Nostalgic",
        "Glamorous",
        "Reflective",
        "Hilarious",
        "Bold",
    ]
    if user_tone in tones:
        return user_tone
    else:
        print("Tone", user_tone, "is not in the list.")
        return 0

def generate_prompt_object(input_value, user_tone):
    """
    Generate a prompt for GPT-3 based on the input value
    and user-specified tone for an object caption.

    Args:
        input_value (str or dict or list): The input value representing the object(s).
        user_tone (str): The desired tone for the caption.

    Returns:
        str: The generated prompt for GPT-3.
    """
    object_name = process_input(input_value)
    tone = check_tone(user_tone)
    gpt_prompt = (
        "Create a "
        + str(tone)
        + " caption and emoji for an image containing "
        + object_name
    )
    return gpt_prompt

def generate_prompt_color(input_value, user_tone):
    """
    Generate a prompt for GPT-3 based on the input value
    and user-specified tone for a color caption.

    Args:
        input_value (str): The input value representing the color.
        user_tone (str): The desired tone for the caption.

    Returns:
        str: The generated prompt for GPT-3.
    """
    color_name = process_input(input_value)
    tone = check_tone(user_tone)
    gpt_prompt = (
        "Create a " + str(tone) + " caption for an image with the color " + color_name
    )
    return gpt_prompt

def generate_gpt3_response(prompt, print_output=False):
    """
    Generate a response from GPT-3 based on the given prompt.

    Args:
        prompt (str): The input prompt for GPT-3.
        print_output (bool, optional): Whether to print the generated output. Default is False.

    Returns:
        str: The generated text response from GPT-3.
    """
    completions = openai.Completion.create(
        engine="text-davinci-003",
        temperature=0.7,
        prompt=prompt,
        max_tokens=40,
        n=1,
        stop=None,
    )
    return completions.choices[0].text

def hugging_face_model(image_path):
    """
    Use the Hugging Face API for image captioning.

    Args:
        image_path (str): The file path to the image.

    Returns:
        dict: The JSON response from the Hugging Face API.
    """
    with open(image_path, "rb") as fname:
        data = fname.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def extract_keywords(sentence):
    """
    Extract keywords (nouns) from a given sentence using spaCy.

    Args:
        sentence (str): The input sentence.

    Returns:
        list: A list of extracted keywords (nouns) from the sentence.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    pos_tags = ["NOUN"]
    keywords = [token.text for token in doc if token.pos_ in pos_tags]
    return keywords

def generate_hugging_face_prompt(keywords, user_tone):
    """
    Generate a prompt for Hugging Face API based on keywords and user tone.

    Args:
        keywords (list): A list of keywords (nouns) extracted from a sentence.
        user_tone (str): The desired tone for the caption.

    Returns:
        str: The generated prompt for the Hugging Face API.
    """
    prompt_template = (
        "Create a " + str(user_tone) + " caption and emoji for an image containing "
    )
    if len(keywords) == 1:
        prompt_template += keywords[0]
    elif len(keywords) == 2:
        prompt_template += f"{keywords[0]} and {keywords[1]}"
    else:
        prompt_template += ", ".join(keywords[:-1]) + f", and {keywords[-1]}"
    return prompt_template + "."

def generate_hashtags_object(input_value):
    """
    Generate hashtags for an image containing the specified object using GPT-3.

    Args:
        input_value (str): The name of the object.

    Returns:
        list: A list of generated hashtags.
    """
    variable = process_input(input_value)
    if variable:
        gpt_prompt = "Generate 5 hashtags for an image containing " + variable
        gpt_output = generate_gpt3_response(gpt_prompt)
        hashtags = re.findall(r"#\w+", gpt_output)
        return hashtags
    else:
        print("Invalid input value!")
        return []

def generate_hashtags_color(input_value):
    """
    Generate hashtags for an image with the specified color using GPT-3.

    Args:
        input_value (str): The name of the color.

    Returns:
        list: A list of generated hashtags.
    """
    color_name = input_value
    if color_name:
        gpt_prompt = "Generate hashtags for an image with the color " + color_name
        gpt_output = generate_gpt3_response(gpt_prompt)
        return gpt_output
    else:
        print("Invalid color name!")
        return []
    
def fetch_reviews_from_database():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute("SELECT rating, comment FROM reviews")
    reviews = cursor.fetchall()
    conn.close()
    return reviews

# Load the YOLO model
model = load_model("yolov8n.pt")

# Set up OpenAI API credentials
openai.organization = "eneter organization key"
openai.api_key = "enter api key"

# Set up HuggingFace API credentials
API_URL = (
    "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
)
headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxx "}

merged_caption = ""

@app.route("/", methods=["GET", "POST"])
def simply():
    """
    Handle the initial rendering of the main page and POST redirection.

    If a POST request is received, this function redirects to the "/about" route.
    Otherwise, it renders the "index.html" template.

    Returns:
        A redirection to the "/about" route if a POST request is received.
        Otherwise, renders the "index.html" template.
    """
    if request.method == "POST":

        return redirect("/about")
    return render_template("index.html")

@app.route("/about", methods=["GET", "POST"])
def upload():
    """
    Handle the upload of an image and process it using various models and APIs.

    If a POST request is received, this function processes the uploaded image:

    Returns:
        Rendered HTML template with processed data.
    """
    result_content = ""
    encoded_img_data = None
    data_to_display = []
    reviews_to_display = []
    if request.method == "POST":
        if 'option' in request.form and 'image' in request.files:
            text = request.form["option"]
            image = request.files["image"]
            #image_data = save_image(image)
          

            filename = "inputimage.jpg"
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(img_path)
            print("Image uploaded and saved successfully")

            with open('uploads/inputimage.jpg', 'rb') as image_file:
                image_data2 = image_file.read()

            # create_table()
            # conn = sqlite3.connect(app.config["DATABASE"])
            # cursor = conn.cursor()
            # cursor.execute(
            #     "INSERT INTO data (text, image) VALUES (?, ?)", (text, image_data)
            # )
            # conn.commit()
            # conn.close()

            if text:
                create_table()
                conn = sqlite3.connect(app.config["DATABASE"])
                cursor = conn.cursor()
                cursor.execute("INSERT INTO data (text, image) VALUES (?, ?)", (text, image_data2))
                conn.commit()
                conn.close()

                # Load the image
                image_path = "uploads/inputimage.jpg"
                img = load_image(image_path)
                #img = Image.open(image)

                # Detect objects in the image and count them
                objects = detect_objects(model, img)
                if len(objects) > 0:
                    result_content = count_objects(objects)
                else:
                    # If no objects are detected, detect the dominant color
                    dominant_color = detect_majority_color(image_path)
                    if dominant_color is not None:
                        # print("\n Dominant Color:", dominant_color)
                        color_name = convert_rgb_to_names(dominant_color)
                        # print(color_name)
                    else:
                        print("No objects and no dominant color detected.")

                # GPT-3 Caption Generation Part
                user_tone = text
                if len(objects) > 0:
                    # Process the object count into a suitable input value for GPT-3
                    input_value = ast.literal_eval("{" + result_content + "}")
                    try:
                        variable = process_input(input_value)
                    except (SyntaxError, NameError):
                        print("Invalid input value!")

                    if variable and user_tone:
                        # Generate a prompt for GPT-3 based on the object and tone
                        gpt_prompt = generate_prompt_object(variable, user_tone)
                        # Generate a caption using GPT-3
                        gpt_output = generate_gpt3_response(gpt_prompt, True)
                        gpt_output = gpt_output.strip()
                        gpt_output = gpt_output.lstrip("\n")
                        caption = gpt_output.replace('"', "")
                        merged_caption = caption
                        print(caption)

                    else:
                        print("Both input value and user tone must be provided correctly.")
                else:
                    # Get the color name and user-specified tone for the caption
                    color_name = convert_rgb_to_names(dominant_color)

                    if color_name and user_tone:
                        # Generate a prompt for GPT-3 based on the color and tone
                        gpt_prompt = generate_prompt_color(color_name, user_tone)
                        print("\n Color Prompt:", gpt_prompt)
                        # Generate a caption using GPT-3
                        gpt_output = generate_gpt3_response(gpt_prompt, True)
                        gpt_output = gpt_output.strip()
                        gpt_output = gpt_output.lstrip("\n")
                        caption = gpt_output.replace('"', "")
                        merged_caption = caption
                        print("\n Color Caption:", caption)
                    else:
                        print("Both color name and user tone must be provided correctly.")

                # Using hugging face api
                hf_output = hugging_face_model(image_path)
                generated_text = hf_output[0]["generated_text"]
                keywords = extract_keywords(generated_text)
                keywords = extract_keywords(generated_text)

                # Generate a prompt using the extracted keywords
                hf_prompt = generate_hugging_face_prompt(keywords, user_tone)
                hf_response = generate_gpt3_response(hf_prompt)
                hf_response = hf_response.strip()
                hf_response = hf_response.lstrip("\n")
                hfcaption = hf_response.replace('"', "")
                merged_caption = merged_caption + "\n" + hfcaption
                print(hfcaption)

                hf_hashtags = generate_hashtags_object(keywords)
                merged_caption = merged_caption + "\n" + ", ".join(hf_hashtags)
                print(hf_hashtags)

                result_content = merged_caption

            with open('uploads/inputimage.jpg', 'rb') as image_file:
                image_data2 = image_file.read()

            encoded_img_data = base64.b64encode(image_data2).decode("utf-8")


            conn = sqlite3.connect(app.config["DATABASE"])
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM data")
            data_rows = cursor.fetchall()
            conn.close()

            # Prepare data to be displayed in the HTML template
            data_to_display = [row[0] for row in data_rows]

            return render_template('about.html',
                       caption=result_content,
                       img_data=encoded_img_data,
                       img_width=400,
                       img_height=300,
                       data=data_to_display)

        elif 'comment' in request.form or 'rating' in request.form:
            create_review_table()
            conn = sqlite3.connect(app.config['DATABASE'])
            cursor = conn.cursor()

            # Insert the review data into the new reviews table
            rating = request.form.get('rating')
            comment = request.form.get('comment')
            cursor.execute("INSERT INTO reviews (rating, comment) VALUES (?, ?)", (rating, comment))
            conn.commit()
            conn.close()

            return redirect('/about')
    
    else:
        create_review_table()
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute("SELECT rating, comment FROM reviews")
        reviews_data = cursor.fetchall()
        conn.close()
        print(reviews_data)
        return render_template('about.html',reviews=reviews_data)
        
    return render_template('about.html',
                           caption=result_content,
                           img_data=encoded_img_data,
                           img_width=400,
                           img_height=300,
                           data=data_to_display
        
                           )

if __name__ == '__main__':
    app.run(debug=True)
