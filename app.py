from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import base64
from io import BytesIO
import requests
import numpy as np
import cv2
import pickle
from keras.models import load_model
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

classes = pickle.load(open("classes.pkl", "rb"))
scientific_names = pickle.load(open("scientific_names.pkl", "rb"))
model = load_model("bird_classifier_fin3.h5")
key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
CORS(app)
@app.route("/classify", methods=["POST"])
def image_check():
    url = request.get_json()['image']
    # Grabbing image data from base64 string or URL
    try:
        # Base64 DATA
        if "data:image/jpeg;base64," in url:
            base_string = url.replace("data:image/jpeg;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img)).convert("RGB")
            img = np.array(img)
            img = cv2.resize(img, (224, 224))

        # Base64 DATA
        elif "data:image/png;base64," in url:
            base_string = url.replace("data:image/png;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img)).convert("RGB")
            img = np.array(img)
            img = cv2.resize(img, (224, 224))

        # Regular URL Form DATA
        else:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = np.array(img)
            img = cv2.resize(img, (224, 224))
    

    # ----- SECTION 3 -----    
        result = "Image has been succesfully sent to the server."
    except Exception as e:
        result = "Error: " + str(e)
        
    predictions = model.predict(img[None, ...])
    index = np.argmax(predictions)
    prompt = PromptTemplate.from_template("You are answering questions about bird species. Tell me two truly random facts about a {bird}, keep the response to less than 150 words and don't number your responses. Do not include any special characters. Try to make them as interesting as possible, and actually random (I keep getting the same responses, over and over).")
    chat = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=key)
    llm_chain = LLMChain(prompt=prompt, llm=chat)
    response = llm_chain.invoke(classes[index])
    return jsonify({"result": classes[index], "scientific_name": scientific_names[index], "confidence": str(round(predictions[0][index] * 100, 2)) + "%", "fact": response['text']})