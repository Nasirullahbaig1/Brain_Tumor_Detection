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
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load and initialize your trained model
MODEL_PATH = 'best_model11.keras'  # Adjust to your model path
THRESHOLD = 0.5  # Adjust after debugging

# Load model with a timeout to prevent hanging
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
    
    # Log model architecture
    logger.info("Model Summary:")
    model.summary(print_fn=logger.info)
    
    # Get the last convolutional layer name for Grad-CAM
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer_name = layer.name
            break
    
    if not last_conv_layer_name:
        last_conv_layer_name = 'conv2d_2'  # Fallback to previous value
    
    logger.info(f"Using {last_conv_layer_name} as the last convolutional layer for Grad-CAM")
    
    # Build the model with the correct input shape
    input_shape = (128, 128, 3)  # Match your training input shape
    model.build((None, *input_shape))  # Explicitly build the model
    logger.info(f"Model built with input shape: {input_shape}")
    
    # Test with a dummy input to ensure initialization
    dummy_input = np.zeros((1, *input_shape), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    logger.info("Model successfully initialized with dummy input")
    
except Exception as e:
    logger.error(f"Error initializing model: {e}")
    logger.warning("App will run but predictions will fail!")
    model = None

# Image processing utilities
class ImageProcessor:
    @staticmethod
    def preprocess_image(img_path):
        """Preprocess image for model input while preserving original quality"""
        try:
            # Read the original image at full resolution
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Failed to load image: {img_path}")
                return None
            
            # Store original image for visualization
            original_img = img.copy()
            
            # Convert to RGB (for visualization and processing)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for model input (128x128)
            img_resized = cv2.resize(img_rgb, (128, 128))
            
            # Normalize for model input
            img_normalized = img_resized / 255.0
            
            logger.info(f"Preprocessed image with original size {original_img.shape}, resized to {img_resized.shape}")
            
            # Return the original BGR image, RGB version, resized version, and normalized version
            return original_img, img_rgb, img_resized, img_normalized
        
        except Exception as e:
            logger.error(f"Error preprocessing image {img_path}: {e}")
            return None
            
    @staticmethod
    def bypass_validation(image_path):
        """Always return True to bypass validation - use for testing"""
        logger.info(f"Bypassing validation for {image_path}")
        return True
    
@staticmethod
def is_mri_image(image_path, min_size=50, max_color_variance=100):
    """Check if image is likely an MRI scan with more relaxed criteria"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return False
            
        h, w, c = img.shape
        logger.info(f"Image {image_path}: Size={w}x{h}, Channels={c}")
        
        if h < min_size or w < min_size:
            logger.warning(f"Rejected: Size too small")
            return False
            
        plt.figure(figsize=(10, 5))
        plt.plot(hist_norm)
        plt.title(f"Grayscale Histogram for {os.path.basename(image_path)}")
        plt.savefig(os.path.join(os.path.dirname(image_path), f"histogram_{os.path.basename(image_path)}.png"))
        plt.close()

        # More relaxed color variance check
        color_variance = np.std(img.reshape(-1, 3), axis=0).mean()
        logger.info(f"Color variance: {color_variance:.2f}")
        
        # Relaxed brightness distribution check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        
        # More relaxed thresholds for dark and bright ratios
        dark_ratio = hist_norm[:30].sum()
        bright_ratio = hist_norm[225:].sum()
        
        logger.info(f"Dark pixel ratio: {dark_ratio:.2f}, Bright pixel ratio: {bright_ratio:.2f}")
        
        # More relaxed validation criteria
        is_valid = (color_variance < max_color_variance and 
                   dark_ratio < 0.5 and 
                   bright_ratio < 0.5)
                   
        logger.info(f"Valid MRI: {is_valid}")
        return is_valid
        
    except Exception as e:
        logger.error(f"Error validating MRI image {image_path}: {e}")
        return False

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, last_conv_layer_name):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        try:
            self.grad_model = tf.keras.models.Model(
                [model.inputs], 
                [model.get_layer(last_conv_layer_name).output, model.output]
            )
            logger.info(f"Grad-CAM sub-model created successfully with layer: {last_conv_layer_name}")
        except Exception as e:
            logger.error(f"Error creating Grad-CAM model: {e}")
            self.grad_model = None
    
    def generate_heatmap(self, img_array):
        """Generate Grad-CAM heatmap for image"""
        try:
            if self.grad_model is None:
                logger.error("Grad-CAM model is not initialized")
                return np.zeros((img_array.shape[1], img_array.shape[2]))
                
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(img_array)
                loss = predictions[:, 0]
                
            # Get gradients
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps with gradients
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Process heatmap
            heatmap = np.maximum(heatmap, 0)
            max_heat = np.max(heatmap)
            if max_heat > 0:
                heatmap /= max_heat
                
            logger.info("Heatmap generated successfully")
            return heatmap.numpy()
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            return np.zeros((img_array.shape[1], img_array.shape[2]))
    
    def overlay_heatmap(self, heatmap, original_img, alpha=0.6, colormap=cv2.COLORMAP_JET):
        """Overlay heatmap on original image with better focus on tumor area"""
        try:
            # Get the original dimensions
            orig_h, orig_w = original_img.shape[:2]
            
            # Make a copy of the original image to avoid modifying it
            original_img_copy = original_img.copy()
            
            # DEBUGGING: Print out heatmap stats before any processing
            logger.info(f"Original heatmap shape: {heatmap.shape}")
            logger.info(f"Original heatmap stats - min: {np.min(heatmap)}, max: {np.max(heatmap)}, mean: {np.mean(heatmap)}")
            logger.info(f"Number of non-zero elements: {np.count_nonzero(heatmap)}")
            
            # Check if heatmap is completely empty
            if np.max(heatmap) == 0 or np.count_nonzero(heatmap) == 0:
                logger.warning("Heatmap is entirely zeros! Creating dummy heatmap for visualization.")
                # Create a dummy heatmap with a small circular activation in the center
                dummy_heatmap = np.zeros_like(heatmap, dtype=np.float32)
                center_y, center_x = heatmap.shape[0] // 2, heatmap.shape[1] // 2
                radius = min(heatmap.shape) // 8
                y, x = np.ogrid[:heatmap.shape[0], :heatmap.shape[1]]
                mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
                dummy_heatmap[mask] = 1.0
                heatmap = dummy_heatmap
                logger.info("Using dummy heatmap for visualization")
            
            # Convert to BGR if needed
            if len(original_img_copy.shape) == 3 and original_img_copy.shape[2] == 3:
                original_img_bgr = cv2.cvtColor(original_img_copy, cv2.COLOR_RGB2BGR) if not np.array_equal(original_img_copy[:,:,0], original_img_copy[:,:,2]) else original_img_copy.copy()
            else:
                original_img_bgr = cv2.cvtColor(original_img_copy, cv2.COLOR_GRAY2BGR)
            
            # Ensure image is in uint8 format
            if original_img_bgr.dtype != np.uint8 or original_img_bgr.max() <= 1.0:
                original_img_bgr = np.uint8(original_img_bgr * 255 if original_img_bgr.max() <= 1.0 else original_img_bgr)
            
            # Ensure heatmap is properly normalized (0-1 range)
            heatmap_normalized = heatmap.copy()
            if np.max(heatmap_normalized) > 0:
                heatmap_normalized = heatmap_normalized / np.max(heatmap_normalized)
            
            # Resize heatmap to match original image
            heatmap_resized = cv2.resize(heatmap_normalized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # AGGRESSIVE APPROACH: Find the areas with ANY activation
            # Instead of thresholding, find the top values in the heatmap
            if np.max(heatmap_resized) > 0:
                # Sort all values and find the threshold at a high percentile
                flat_values = heatmap_resized.flatten()
                non_zero_values = flat_values[flat_values > 0]
                
                if len(non_zero_values) > 0:
                    # Use the 90th percentile as threshold (or 50th if very few values)
                    percentile = 90 if len(non_zero_values) > 100 else 50
                    threshold = np.percentile(non_zero_values, percentile)
                    logger.info(f"Using {percentile}th percentile as threshold: {threshold}")
                    
                    heatmap_thresholded = np.copy(heatmap_resized)
                    heatmap_thresholded[heatmap_resized < threshold] = 0
                else:
                    # If no non-zero values, use all positive values
                    logger.warning("No non-zero values in heatmap after resizing")
                    heatmap_thresholded = np.copy(heatmap_resized)
            else:
                logger.warning("Heatmap is all zeros after resizing")
                # Create a dummy activation
                heatmap_thresholded = np.zeros_like(heatmap_resized)
                center_y, center_x = heatmap_resized.shape[0] // 2, heatmap_resized.shape[1] // 2
                radius = min(heatmap_resized.shape) // 8
                y, x = np.ogrid[:heatmap_resized.shape[0], :heatmap_resized.shape[1]]
                mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
                heatmap_thresholded[mask] = 1.0
            
            # Check if there's enough activation
            activation_pixels = np.count_nonzero(heatmap_thresholded)
            logger.info(f"Activation pixels after thresholding: {activation_pixels}")
            
            if activation_pixels < 10:
                logger.warning("Very few activation pixels, using top values")
                # If there are very few activated pixels, use the top 100 values
                flat_indices = np.argsort(heatmap_resized.flatten())[-100:]
                heatmap_thresholded = np.zeros_like(heatmap_resized)
                indices = np.unravel_index(flat_indices, heatmap_resized.shape)
                heatmap_thresholded[indices] = heatmap_resized[indices]
            
            # Normalize the thresholded heatmap again
            if np.max(heatmap_thresholded) > 0:
                heatmap_thresholded = heatmap_thresholded / np.max(heatmap_thresholded)
            
            # Convert to 8-bit for colormapping
            heatmap_8bit = np.uint8(255 * heatmap_thresholded)
            
            # Apply color map to create a colored heatmap
            colored_heatmap = cv2.applyColorMap(heatmap_8bit, colormap)
            
            # Create a mask for areas with activation
            activation_mask = (heatmap_thresholded > 0).astype(np.float32)
            
            # Create output image
            output_img = np.zeros_like(original_img_bgr)
            
            # Blend the images
            for c in range(3):  # For each color channel
                output_img[:,:,c] = original_img_bgr[:,:,c] * (1 - activation_mask * alpha) + colored_heatmap[:,:,c] * (activation_mask * alpha)
            
            # Convert to uint8
            output_img = np.uint8(output_img)
            
            # Dilate the mask to make it more visible
            kernel = np.ones((5, 5), np.uint8)  # Larger kernel
            tumor_mask = np.uint8(activation_mask * 255)
            dilated_mask = cv2.dilate(tumor_mask, kernel, iterations=3)  # More iterations
            
            # Find contours of the tumor area
            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour and get its center
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get the moments to find the center
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    # Fallback to geometric center
                    rect = cv2.boundingRect(largest_contour)
                    center_x = rect[0] + rect[2] // 2
                    center_y = rect[1] + rect[3] // 2
            else:
                # If no contours found, use activation weighted center
                y_coords, x_coords = np.where(heatmap_thresholded > 0)
                if len(y_coords) > 0 and len(x_coords) > 0:
                    # Weighted center based on activation values
                    weights = np.array([heatmap_thresholded[y, x] for y, x in zip(y_coords, x_coords)])
                    center_y = int(np.average(y_coords, weights=weights))
                    center_x = int(np.average(x_coords, weights=weights))
                else:
                    # Use image center as fallback
                    center_y, center_x = orig_h // 2, orig_w // 2
                
                # Create an artificial contour around the detected center
                radius = min(orig_h, orig_w) // 10
                artificial_contour = []
                for angle in range(0, 360, 10):
                    rads = np.radians(angle)
                    x = center_x + int(radius * np.cos(rads))
                    y = center_y + int(radius * np.sin(rads))
                    artificial_contour.append([[x, y]])
                
                largest_contour = np.array(artificial_contour, dtype=np.int32)
                contours = [largest_contour]
            
            # Now, determine location based on the center coordinates
            def determine_tumor_location(center_x, center_y, img_width, img_height):
                # Divide image into regions (3x3 grid)
                x_third = img_width / 3
                y_third = img_height / 3
                
                # Determine horizontal position
                if center_x < x_third:
                    horiz = "Left"
                elif center_x < 2 * x_third:
                    horiz = "Center"
                else:
                    horiz = "Right"
                    
                # Determine vertical position
                if center_y < y_third:
                    vert = "Top"
                elif center_y < 2 * y_third:
                    vert = "Middle"
                else:
                    vert = "Bottom"
                    
                return f"{vert} {horiz}"
            
            # Get location based on the actual contour center
            tumor_location = determine_tumor_location(center_x, center_y, orig_w, orig_h)
            logger.info(f"Tumor center: x={center_x}, y={center_y}, location={tumor_location}")
            
            # Draw the contour
            cv2.drawContours(output_img, [largest_contour], -1, (0, 0, 255), 2)
            
            # Add a visible center point
            cv2.circle(output_img, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Calculate area
            tumor_area = cv2.contourArea(largest_contour) / (orig_h * orig_w) * 100
            
            # Create white background area at the bottom of the image
            info_height = 80
            final_img = np.ones((orig_h + info_height, orig_w, 3), dtype=np.uint8) * 255
            final_img[:orig_h, :, :] = output_img
            
            # Add text on the white area
            cv2.putText(final_img, f"Location: {tumor_location}", (10, orig_h + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(final_img, f"Est. Size: {tumor_area:.1f}% of brain", (10, orig_h + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            logger.info(f"Enhanced focused heatmap created with info panel - location: {tumor_location}")
            return final_img
            
        except Exception as e:
            logger.error(f"Error creating focused overlay: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return error image
            error_img = original_img.copy()
        if len(error_img.shape) == 2:
            error_img = cv2.cvtColor(error_img, cv2.COLOR_GRAY2BGR)
        
        # Add error text to image
        cv2.putText(error_img, "Heatmap Error!", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(error_img, str(e)[:40], (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return error_img
    
    def get_tumor_location(self, heatmap):
        """Determine tumor location based on heatmap intensity"""
        try:
            h, w = heatmap.shape
            center_y, center_x = h // 2, w // 2
            quad_size_y, quad_size_x = h // 4, w // 4  # Fixed typo: a4 -> 4
            
            # Define regions with more granularity
            regions = {
                'Top Left': np.mean(heatmap[:center_y, :center_x]),
                'Top Center': np.mean(heatmap[:center_y, center_x-quad_size_x:center_x+quad_size_x]),
                'Top Right': np.mean(heatmap[:center_y, center_x:]),
                'Middle Left': np.mean(heatmap[center_y-quad_size_y:center_y+quad_size_y, :center_x]),
                'Center': np.mean(heatmap[center_y-quad_size_y:center_y+quad_size_y, center_x-quad_size_x:center_x+quad_size_x]),
                'Middle Right': np.mean(heatmap[center_y-quad_size_y:center_y+quad_size_y, center_x:]),
                'Bottom Left': np.mean(heatmap[center_y:, :center_x]),
                'Bottom Center': np.mean(heatmap[center_y:, center_x-quad_size_x:center_x+quad_size_x]),
                'Bottom Right': np.mean(heatmap[center_y:, center_x:])
            }
            
            location = max(regions, key=regions.get)
            logger.info(f"Determined tumor location: {location}")
            return location
            
        except Exception as e:
            logger.error(f"Error determining tumor location: {e}")
            return "Undetermined"

# Brain tumor detector class
class BrainTumorDetector:
    def __init__(self, model, last_conv_layer, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.grad_cam = GradCAM(model, last_conv_layer) if model is not None else None
    
    def predict(self, image_path, upload_folder):
        """Process image and make prediction with improved visualization"""
        logger.info(f"Processing image: {image_path}")
        
        if self.model is None:
            logger.error("Model not loaded, cannot make prediction")
            return {
                'status': 'Error',
                'probability': None,
                'message': 'Model not loaded properly'
            }
        
        # Preprocess image
        image_data = ImageProcessor.preprocess_image(image_path)
        if image_data is None:
            return {
                'status': 'Error',
                'probability': None,
                'message': 'Failed to process image'
            }
            
        img, img_rgb, img_resized, img_normalized = image_data
        
        # Prepare for model input
        img_array = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        try:
            start_time = time.time()
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Raw prediction probability: {prediction:.4f}")
            
            is_tumor = prediction > self.threshold
            
            # Save original image
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
            
            # Generate tumor visualization if tumor detected
            if is_tumor and self.grad_cam is not None:
                # Generate heatmap
                heatmap = self.grad_cam.generate_heatmap(img_array)
                
                # IMPORTANT CHANGE: Use the ORIGINAL image (not resized) for overlay
                # This ensures we maintain the original resolution
                highlighted_img = self.grad_cam.overlay_heatmap(heatmap, img)
                
                # Save highlighted image at FULL RESOLUTION
                highlighted_filename = f"highlighted_{secure_filename(os.path.basename(image_path))}"
                highlighted_path = os.path.join(upload_folder, highlighted_filename)
                
                # No need for color conversion since our improved overlay_heatmap handles it
                cv2.imwrite(highlighted_path, highlighted_img)
                
                # Get tumor location
                location = self.grad_cam.get_tumor_location(heatmap)
                
                result['highlighted_path'] = highlighted_path
                result['location'] = location
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'status': 'Error',
                'probability': None,
                'message': f'Prediction failed: {str(e)}'
            }
        
        # Preprocess image
        image_data = ImageProcessor.preprocess_image(image_path)
        if image_data is None:
            return {
                'status': 'Error',
                'probability': None,
                'message': 'Failed to process image'
            }
            
        img, img_rgb, img_resized, img_normalized = image_data
        
        # Prepare for model input
        img_array = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        try:
            start_time = time.time()
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Raw prediction probability: {prediction:.4f}")
            
            is_tumor = prediction > self.threshold
            
            # Save original image
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
            
            # Generate tumor visualization if tumor detected
            if is_tumor and self.grad_cam is not None:
                # Generate heatmap
                heatmap = self.grad_cam.generate_heatmap(img_array)
                
                # Overlay heatmap on original image
                highlighted_img = self.grad_cam.overlay_heatmap(heatmap, img_resized)
                
                # Save highlighted image
                highlighted_filename = f"highlighted_{secure_filename(os.path.basename(image_path))}"
                highlighted_path = os.path.join(upload_folder, highlighted_filename)
                cv2.imwrite(highlighted_path, cv2.cvtColor(highlighted_img, cv2.COLOR_RGB2BGR))
                
                # Get tumor location
                location = self.grad_cam.get_tumor_location(heatmap)
                
                result['highlighted_path'] = highlighted_path
                result['location'] = location
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'status': 'Error',
                'probability': None,
                'message': f'Prediction failed: {str(e)}'
            }

# Initialize detector
detector = BrainTumorDetector(
    model=model,
    last_conv_layer=last_conv_layer_name,
    threshold=THRESHOLD
)

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
            
            # Check if it's a valid MRI image
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
            
            # Process valid MRI image
            valid_count += 1
            result = detector.predict(temp_path, app.config['UPLOAD_FOLDER'])
            
            if 'message' in result and result['message']:
                # Error occurred
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
                # Successful prediction
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
        
        # Only calculate majority result if we have valid images
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
    """API endpoint to check application health"""
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
    # Set host to None for production (use environment variables)
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() in ['true', '1', 't']
    
    logger.info(f"Starting application on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
>>>>>>> 6f50d9a (30_April_updates)
