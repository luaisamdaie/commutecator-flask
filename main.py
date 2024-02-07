from flask import Flask, request, jsonify
from keras.models import load_model
from cv2 import imdecode, resize
import numpy as np


app = Flask(__name__)

@app.before_first_request
def load_model():
    global model
    model = load_model("/home/luaihani/commutecator/commutecator.h5", compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


IMAGE_SIZE = 48
LABEL_MAPPING = [
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "space", "del", "nothing",
]


def preprocess_image(image):
    image = np.asarray(bytearray(image.read()), dtype=np.uint8)
    image = imdecode(image, 1)

    image = resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)

    return image


def get_prediction(image):
    prediction = model.predict(image)
    prediction = np.argmax(prediction, axis=1)
    prediction[0] = LABEL_MAPPING[prediction[0]]
    return prediction
    # return LABEL_MAPPING[prediction]


@app.route("/")
def index():
    return "Hello from Flask!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = request.files["image"].stream
        image = preprocess_image(image)
        prediction = get_prediction(image)
        return jsonify({"prediction": prediction})

    except Exception as e:
        return {'error': str(e)}


# app.run()