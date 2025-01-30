from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = load_model('best_model9.keras')

# Define the upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to preprocess the image for prediction
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file part in the request")
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        print("No file selected")
        return redirect(url_for('home'))

    print(f"File uploaded: {file.filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    print(f"File saved at: {file_path}")

    img_array = preprocess_image(file_path)
    if img_array is None:
        print("Error preprocessing image")
        return redirect(url_for('home'))

    prediction = model.predict(img_array)
    print(f"Prediction: {prediction}")
    result = "Tumor Positive" if prediction[0][0] > 0.3 else "Tumor Negative"
    print(f"Result: {result}")

    return render_template('result.html', result=result, image_file=file.filename)

if __name__ == '__main__':
    app.run(debug=True)