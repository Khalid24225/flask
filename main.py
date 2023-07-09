from flask import Flask, jsonify, request
import joblib
from keras.models import load_model
import tensorflow as tf
from PIL import Image
from flask_cors import CORS, cross_origin
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
cors = CORS(app)

diabeticML = tf.keras.models.load_model('C:/Users/zas/Desktop/ahmed/Through-ur-eyes/ML/Diabetic.h5')
anemiaML = tf.keras.models.load_model('C:/Users/zas/Desktop/ahmed/Through-ur-eyes/ML/model2.h5')
hypertensionML = tf.keras.models.load_model('C:/Users/zas/Desktop/ahmed/Through-ur-eyes/ML/model3.pkl')

diabeticCategories = [
    ['No DR', 'tips 1'],
    ['Mild', 'tips 2'], 
    ['Moderate', 'tips 3'], 
    ['Severe', 'tips 4'],
    ['Proliferative DR', 'tips 5'],
]

anemiaCategories = [
    ['Anemia', 'tips 1'],
    ['Non Anemia', 'tips 2'], 
]
# <h2 id="health-ratio">12.7 <sub>gdl</sub><sup>-1</sup></h2>
hypertensionCategories = [
    ['Normal', 'tips 1'],
    ['Grade 1', 'tips 2'], 
    ['Grade 2', 'tips 3'], 
    ['Grade 3', 'tips 4'],
    ['Grade 4', 'tips 5'],
]


@app.route('/predict/diabetic', methods=['POST'])
def diabetic():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.resize((150, 150))  # Resize the image to match the input shape of the model
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension to match the model's input shape

    prediction = diabeticML.predict(img_array)[0]
    index_ = np.argmax(prediction)

    return jsonify({
        'health': diabeticCategories[index_][0],
        'tips': diabeticCategories[index_][1],
    })


@app.route('/predict/anemia', methods=['POST'])
def anemia():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.resize((150, 150))  # Resize the image to match the input shape of the model
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension to match the model's input shape

    prediction = anemiaML.predict(img_array)[0]
    index_ = np.argmax(prediction)

    return jsonify({
        'health': anemiaCategories[index_][0],
        'tips': anemiaCategories[index_][1],
    })


@app.route('/predict/hypertension', methods=['POST'])
def hypertension():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.resize((150, 150))  # Resize the image to match the input shape of the model
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension to match the model's input shape

    prediction = hypertensionML.predict(img_array)[0]
    index_ = np.argmax(prediction)

    return jsonify({
        'health': hypertensionCategories[index_][0],
        'tips': hypertensionCategories[index_][1],
    })

if __name__ == '__main__':
    app.run()
