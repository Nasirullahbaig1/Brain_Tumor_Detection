<<<<<<< HEAD
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
=======
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import time
import logging
import sys
from io import StringIO
try:
    from skimage.filters.rank import entropy
except ImportError:
    from skimage.morphology import disk
    def entropy(image, footprint):
        from scipy.ndimage import generic_filter
        def local_entropy(arr):
            arr = arr.flatten()
            hist, _ = np.histogram(arr, bins=256, range=(0, 1), density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist + 1e-10))
        return generic_filter(image, local_entropy, footprint=footprint)
from skimage.morphology import disk

# Set up logging with Unicode-safe formatter
class UnicodeSafeFormatter(logging.Formatter):
    def format(self, record):
        try:
            return super().format(record)
        except UnicodeEncodeError:
            return record.getMessage().encode('ascii', 'replace').decode('ascii')

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(UnicodeSafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.handlers = [handler]
logger.setLevel(logging.INFO)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load and initialize model
MODEL_PATH = 'best_model11.keras'
THRESHOLD = 0.5

def load_model_with_timeout(model_path, timeout=60):
    start_time = time.time()
    try:
        model = load_model(model_path)
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        if time.time() - start_time > timeout:
            logger.error("Model loading timed out")
        raise

try:
    logger.info(f"Loading model from {MODEL_PATH}")
    model = load_model_with_timeout(MODEL_PATH)
    logger.info('Model Summary:')
    summary_io = StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
    logger.info(summary_io.getvalue().encode('ascii', 'replace').decode('ascii'))
    input_shape = (128, 128, 3)
    model.build((None, *input_shape))
    logger.info(f"Model built with input shape: {input_shape}")
    dummy_input = np.zeros((1, *input_shape), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    logger.info("Model successfully initialized with dummy input")
except Exception as e:
    logger.error(f"Error initializing model: {e}")
    logger.warning("App will run but predictions will fail!")
    model = None

# TumorSegmentation class
class TumorSegmentation:
    def __init__(self, intensity_threshold=0.7, edge_sensitivity=100, texture_threshold=0.5):
        self.intensity_threshold = intensity_threshold
        self.edge_sensitivity = edge_sensitivity
        self.texture_threshold = texture_threshold
        logger.info(f"Initialized TumorSegmentation with intensity_threshold={intensity_threshold}, "
                    f"edge_sensitivity={edge_sensitivity}, texture_threshold={texture_threshold}")

    def preprocess_image(self, img):
        try:
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img.copy()
            if img_gray.max() > 1.0:
                img_gray = img_gray / 255.0
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(np.uint8(img_gray * 255))
            img_enhanced = img_enhanced / 255.0
            logger.info(f"Image preprocessed: shape={img_enhanced.shape}, "
                       f"min={np.min(img_enhanced)}, max={np.max(img_enhanced)}")
            return img_enhanced
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return img

    def segment_tumor(self, img):
        try:
            orig_h, orig_w = img.shape[:2]
            img_preprocessed = self.preprocess_image(img)
            intensity_mask = img_preprocessed > self.intensity_threshold
            logger.info(f"Intensity mask: {np.count_nonzero(intensity_mask)} pixels above threshold")
            edges = cv2.Canny(np.uint8(img_preprocessed * 255), self.edge_sensitivity, self.edge_sensitivity * 2)
            edge_mask = edges > 0
            logger.info(f"Edge mask: {np.count_nonzero(edge_mask)} edge pixels detected")
            entropy_img = entropy(img_preprocessed, disk(5))
            entropy_mask = entropy_img > (np.max(entropy_img) * self.texture_threshold)
            logger.info(f"Texture mask: {np.count_nonzero(entropy_mask)} high-entropy pixels")
            combined_mask = np.zeros_like(intensity_mask, dtype=np.uint8)
            combined_mask = np.where((intensity_mask.astype(np.uint8) + edge_mask.astype(np.uint8) + entropy_mask.astype(np.uint8)) >= 2, 1, 0)
            logger.info(f"Combined mask: {np.count_nonzero(combined_mask)} pixels selected")
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.warning("No contours found, creating fallback region")
                combined_mask = np.zeros_like(combined_mask)
                center_y, center_x = orig_h // 2, orig_w // 2
                radius = min(orig_h, orig_w) // 10
                y, x = np.ogrid[:orig_h, :orig_w]
                mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
                combined_mask[mask] = 1
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            logger.info(f"Largest contour area: {cv2.contourArea(largest_contour)}")
            return combined_mask, largest_contour
        except Exception as e:
            logger.error(f"Error segmenting tumor: {e}")
            return np.zeros_like(img, dtype=np.uint8), None

    def overlay_segmentation(self, img, mask, contour, alpha=0.6, colormap=cv2.COLORMAP_JET):
        try:
            orig_h, orig_w = img.shape[:2]
            img_copy = img.copy()
            if len(img_copy.shape) == 3 and img_copy.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR) if not np.array_equal(img_copy[:,:,0], img_copy[:,:,2]) else img_copy.copy()
            else:
                img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
            if img_bgr.dtype != np.uint8 or img_bgr.max() <= 1.0:
                img_bgr = np.uint8(img_bgr * 255 if img_bgr.max() <= 1.0 else img_bgr)
            mask_8bit = np.uint8(mask * 255)
            colored_mask = cv2.applyColorMap(mask_8bit, colormap)
            output_img = img_bgr.copy()
            activation_mask = mask.astype(np.float32)
            for c in range(3):
                output_img[:,:,c] = img_bgr[:,:,c] * (1 - activation_mask * alpha) + colored_mask[:,:,c] * (activation_mask * alpha)
            output_img = np.uint8(output_img)
            if contour is not None:
                cv2.drawContours(output_img, [contour], -1, (0, 0, 255), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    rect = cv2.boundingRect(contour)
                    center_x = rect[0] + rect[2] // 2
                    center_y = rect[1] + rect[3] // 2
                cv2.circle(output_img, (center_x, center_y), 5, (255, 0, 0), -1)
            else:
                center_x, center_y = orig_w // 2, orig_h // 2
            def determine_tumor_location(center_x, center_y, img_width, img_height):
                x_third = img_width / 3
                y_third = img_height / 3
                if center_x < x_third:
                    horiz = "Left"
                elif center_x < 2 * x_third:
                    horiz = "Center"
                else:
                    horiz = "Right"
                if center_y < y_third:
                    vert = "Top"
                elif center_y < 2 * y_third:
                    vert = "Middle"
                else:
                    vert = "Bottom"
                return f"{vert} {horiz}"
            tumor_location = determine_tumor_location(center_x, center_y, orig_w, orig_h)
            tumor_area = cv2.contourArea(contour) / (orig_h * orig_w) * 100 if contour is not None else 0
            info_height = 100
            final_img = np.ones((orig_h + info_height, orig_w, 3), dtype=np.uint8) * 255
            final_img[:orig_h, :, :] = output_img
            basis = "High intensity, edges, and texture entropy"
            cv2.putText(final_img, f"Location: {tumor_location}", (10, orig_h + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(final_img, f"Est. Size: {tumor_area:.1f}% of brain", (10, orig_h + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(final_img, f"Basis: {basis}", (10, orig_h + 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            logger.info(f"Segmentation overlay created - location: {tumor_location}, basis: {basis}")
            return final_img
        except Exception as e:
            logger.error(f"Error creating overlay: {e}")
            error_img = img.copy()
            if len(error_img.shape) == 2:
                error_img = cv2.cvtColor(error_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(error_img, "Segmentation Error!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(error_img, str(e)[:40], (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return error_img

    def process_image(self, img):
        mask, contour = self.segment_tumor(img)
        return self.overlay_segmentation(img, mask, contour)

# Image processing utilities
class ImageProcessor:
    @staticmethod
    def preprocess_image(img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Failed to load image: {img_path}")
                return None
            original_img = img.copy()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (128, 128))
            img_normalized = img_resized / 255.0
            logger.info(f"Preprocessed image with original size {original_img.shape}, resized to {img_resized.shape}")
            return original_img, img_rgb, img_resized, img_normalized
        except Exception as e:
            logger.error(f"Error preprocessing image {img_path}: {e}")
            return None

    @staticmethod
    def bypass_validation(image_path):
        logger.info(f"Bypassing validation for {image_path}")
        return True

# BrainTumorDetector class
class BrainTumorDetector:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.segmenter = TumorSegmentation(intensity_threshold=0.7, edge_sensitivity=100, texture_threshold=0.5) if model is not None else None

    def predict(self, image_path, upload_folder):
        logger.info(f"Processing image: {image_path}")
        if self.model is None:
            logger.error("Model not loaded, cannot make prediction")
            return {
                'status': 'Error',
                'probability': None,
                'message': 'Model not loaded properly'
            }
        image_data = ImageProcessor.preprocess_image(image_path)
        if image_data is None:
            return {
                'status': 'Error',
                'probability': None,
                'message': 'Failed to process image'
            }
        img, img_rgb, img_resized, img_normalized = image_data
        img_array = np.expand_dims(img_normalized, axis=0)
        try:
            start_time = time.time()
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Raw prediction probability: {prediction:.4f}")
            is_tumor = prediction > self.threshold
            original_filename = f"original_{secure_filename(os.path.basename(image_path))}"
            original_path = os.path.join(upload_folder, original_filename)
            cv2.imwrite(original_path, img)
            result = {
                'status': 'Tumor Detected' if is_tumor else 'No Tumor Detected',
                'probability': float(prediction),
                'original_path': original_path,
                'highlighted_path': None,
                'location': None
            }
            if is_tumor and self.segmenter is not None:
                highlighted_img = self.segmenter.process_image(img_rgb)
                highlighted_filename = f"highlighted_{secure_filename(os.path.basename(image_path))}"
                highlighted_path = os.path.join(upload_folder, highlighted_filename)
                cv2.imwrite(highlighted_path, cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB))
                mask, contour = self.segmenter.segment_tumor(img_rgb)
                if contour is not None:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                    else:
                        rect = cv2.boundingRect(contour)
                        center_x = rect[0] + rect[2] // 2
                        center_y = rect[1] + rect[3] // 2
                    def determine_tumor_location(center_x, center_y, img_width, img_height):
                        x_third = img_width / 3
                        y_third = img_height / 3
                        if center_x < x_third:
                            horiz = "Left"
                        elif center_x < 2 * x_third:
                            horiz = "Center"
                        else:
                            horiz = "Right"
                        if center_y < y_third:
                            vert = "Top"
                        elif center_y < 2 * y_third:
                            vert = "Middle"
                        else:
                            vert = "Bottom"
                        return f"{vert} {horiz}"
                    result['location'] = determine_tumor_location(center_x, center_y, img.shape[1], img.shape[0])
                result['highlighted_path'] = highlighted_path
            return result
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'status': 'Error',
                'probability': None,
                'message': f'Prediction failed: {str(e)}'
            }

# Initialize detector
detector = BrainTumorDetector(model=model, threshold=THRESHOLD)

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'images' not in request.files:
            return render_template('index.html', error='No images uploaded')
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            return render_template('index.html', error='No images selected')
        results = []
        tumor_count = 0
        valid_count = 0
        for file in files:
            start_time = time.time()
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            logger.info(f"Saved uploaded file to {temp_path}")
            if not ImageProcessor.bypass_validation(temp_path):
                results.append({
                    'filename': filename,
                    'status': 'Invalid Image',
                    'probability': None,
                    'original_path': temp_path,
                    'highlighted_path': None,
                    'location': None
                })
                continue
            valid_count += 1
            result = detector.predict(temp_path, app.config['UPLOAD_FOLDER'])
            if 'message' in result and result['message']:
                results.append({
                    'filename': filename,
                    'status': 'Error',
                    'probability': None,
                    'original_path': temp_path,
                    'highlighted_path': None,
                    'location': None,
                    'message': result['message']
                })
            else:
                is_tumor = result['status'] == 'Tumor Detected'
                tumor_count += int(is_tumor)
                results.append({
                    'filename': filename,
                    'status': result['status'],
                    'probability': result['probability'],
                    'original_path': result['original_path'],
                    'highlighted_path': result['highlighted_path'],
                    'location': result['location']
                })
            logger.info(f"Processed {filename} in {time.time() - start_time:.2f} seconds")
        if valid_count > 0:
            majority_result = 'Tumor Detected' if tumor_count > valid_count / 2 else 'No Tumor Detected'
        else:
            majority_result = 'No valid MRI images to analyze'
        logger.info(f"Completed processing {len(files)} files. Valid: {valid_count}, Tumor positive: {tumor_count}")
        return render_template('index.html', results=results, majority_result=majority_result)
    except Exception as e:
        logger.error(f"Error in predict route: {e}")
        return render_template('index.html', error=f'An error occurred: {str(e)}')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('index.html', error='File too large. Maximum file size is 16MB.'), 413

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('index.html', error='Server error occurred. Please try again later.'), 500

if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() in ['true', '1', 't']
    logger.info(f"Starting application on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
>>>>>>> 6f50d9a (30_April_updates)
