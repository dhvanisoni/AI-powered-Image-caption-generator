# SnapSpeak - Image Caption Generator

## About the Project
Welcome to the Image Caption Generator project! This innovative web application leverages cutting-edge AI technologies, including the YOLO (You Only Look Once) model for object detection and a Language Learning Model (LLM) API for generating natural and contextually relevant captions for images. Designed with user experience in mind, The Flask-based web application offers a seamless and interactive user experience, allowing for the effortless upload of images and instant caption generation.

![Project structire](https://github.com/dhvanisoni/AI-powered-Image-caption-generator/blob/main/images/diagram.png)

## Getting Started
To explore the capabilities of the Image Caption Generator, follow these instructions to set up the project on your local environment.

Prerequisites
Make sure you have Python installed on your system. If not, you can download it from python.org.

1. Clone the Repository: Open your terminal and navigate to the directory where you want to clone the project. Then, run the following command to clone the repository:
```bash
 git clone https://github.com/dhvanisoni/AI-powered-Image-Caption-Generator.git
```
2. Navigate to the project directory:
 ```bash
  cd image-caption-generator
```
3. Install Dependencies: Ensure you have Python and pip installed. Create a virtual environment (optional but recommended) and activate it. Then, install the required dependencies.
```bash
  pip install -r requirements.txt
```
4. Run the App in vs code: Start the app by executing the following command:
```bash
  python app.py
```
## Usage
- The interface is intuitive: upload an image, and the system will present the image alongside its AI-generated caption.
- Test with various images to discover the adaptability of our captioning system to different scenarios and objects.
  
## 🚀 Features
- **Advanced Object Detection**: Leverages powerful YOLOv8, the latest iteration of the "You Only Look Once" model, known for its exceptional accuracy and speed in detecting objects within images.
- **Caption Generation**: Integrates with OpenAI GPT-3 and Hugging Face's transformer model's API to craft coherent and contextually relevant captions based on the objects detected.
- **User-Friendly Interface**: Provides an easy-to-navigate web interface built with Flask, allowing users to upload images effortlessly and receive captions in seconds.
- **Performance Optimization**: Engineered for precision, accuracy in object detection and caption relevance, thanks to continuous refinement and the integration of cutting-edge AI models.  

## Tech Stack:
- Programming Language: Python
- Libraries/Frameworks: pandas, NumPy, scikit-learn, TensorFlow, Flask 
- Models and API: Yolov8n model, OpenAI GPT 3 model API, Huggingface model API
- HTML, css and javascript for web application

