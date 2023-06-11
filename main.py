import io
import json
from flask import Flask, request, jsonify
from keras.models import load_model
from tensorflow import keras
from PIL import Image
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Membaca file JSON
with open('deskripsi.json', 'r') as file:
    data = json.load(file)

    # Mengambil objek dari data JSON
    objek = data['deskripsi']

# Load Model
model = load_model("ModelInceptionSnapZoo.h5", compile=False)
model.build((None, 299, 299, 3))

# Classes Label
labels = ['badak', 'gajah', 'harimau', 'jerapah', 'monyet', 'penguin', 'rusa', 'singa', 'ular', 'zebra']
image_size = 150

# Convert Image to Appropriate Format
def format_image(image):
    image = np.array(image.resize((299, 299))) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Predict Image
def predict_image(image):
    prediction = model.predict(image)
    return int(np.argmax(prediction))

@app.route("/predict", methods=["POST"])
def request_prediction():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": True, "message": "No image file to predict."})

        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            tensor_image = format_image(image)
            prediction = predict_image(tensor_image)

            # Mengganti label dengan objek dari JSON
            label = labels[prediction]
            objek_prediksi = objek[label]

            result = { "label": label, "hewan": objek_prediksi}

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": True, "message": str(e)})

    return jsonify({"error": False, "message": "Prediction service online. Try POST method to predict an image file."})

@app.route("/deskripsi", methods=["GET"])
def get_objects():
    return jsonify(objek)

@app.route("/deskripsi/<string:animal>", methods=["GET"])
def get_description(animal):
    if animal in data["deskripsi"]:
        return jsonify(data["deskripsi"][animal])
    else:
        return jsonify({"error": True, "message": "Object not found."}), 404

if __name__ == "__main__":
    app.run(debug=True)
