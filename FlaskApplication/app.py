# import os
# import numpy as np
# from PIL import Image
# import cv2
# from flask import Flask, request, render_template
# from werkzeug.utils import secure_filename
# from tensorflow.keras.models import load_model

# # Load the entire model
# model = load_model('ensemble_model_best(92.3).h5')

# app = Flask(__name__)

# print('Model loaded. Check http://127.0.0.1:5000/')

# def get_className(classNo):
#     if classNo == 0:
#         return "Normal"
#     elif classNo == 1:
#         return "Pneumonia"

# def getResult(img):
#     image = cv2.imread(img)
#     image = Image.fromarray(image, 'RGB')
#     image = image.resize((224, 224))
#     image = np.array(image)
#     input_img = np.expand_dims(image, axis=0) / 255.0  # Ensure normalization
#     result = model.predict(input_img)
#     result01 = np.argmax(result, axis=1)
#     return result01

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def upload():
#     if request.method == 'POST':
#         f = request.files['file']

#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)
#         value = getResult(file_path)
#         result = get_className(value[0])  # Extract the integer from array
#         return result
#     return None

# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=5000,debug=True)


# for render deployment
import os
import numpy as np
from PIL import Image
import cv2
import requests
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# -------------------------------
# Download model from Dropbox if not present
MODEL_URL = "https://www.dropbox.com/scl/fi/l6nm2xh6u7xorbj1d4pz2/ensemble_model_best-92.3.h5?rlkey=a3nan2vmyp19wfe1k7k7p3rll&st=z2idkyh5&dl=1"
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ensemble_model_best(92.3).h5')

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("Download complete.")
# -------------------------------

# Load the model
model = load_model(MODEL_PATH)

app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "Normal"
    elif classNo == 1:
        return "Pneumonia"

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0) / 255.0  # Ensure normalization
    result = model.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return result01

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value[0])  # Extract the integer from array
        return result
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
