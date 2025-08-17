import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash, make_response # Ensure make_response is imported
from werkzeug.utils import secure_filename
import time # To create unique filenames

# --- Configuration ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
MASK_FOLDER = os.path.join('static', 'masks')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MODEL_PATH = 'unet_forgery_detector.keras' # Make sure this matches your saved model file
TARGET_SIZE = (256, 256)
FORGERY_THRESHOLD = 0.01 # Changed threshold to 0.5% # Same threshold used in evaluation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER
# IMPORTANT: Use a real secret key! Generate one using secrets.token_hex(24)
app.secret_key = '7af8df1b037d73c6013ae5d5ccb9676a708e70a06ca44bbe' # Replace with your actual key

# --- Create directories if they don't exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MASK_FOLDER'], exist_ok=True)

# --- Load Model (Do this ONCE at startup) ---
print(f"Loading model from {MODEL_PATH}...")
try:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Uncomment if forcing CPU
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dct(image, block_size=8):
    if image.dtype != np.float32:
         image = image.astype(np.float32)
    height, width = image.shape[:2]
    h_blocks = height // block_size
    w_blocks = width // block_size
    dct_block_image = np.zeros((height, width), np.float32)
    for r in range(h_blocks):
        for c in range(w_blocks):
            row_start, row_end = r * block_size, (r + 1) * block_size
            col_start, col_end = c * block_size, (c + 1) * block_size
            current_block = image[row_start:row_end, col_start:col_end]
            dct_coeffs = cv2.dct(current_block)
            dct_block_image[row_start:row_end, col_start:col_end] = dct_coeffs
    return dct_block_image

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None
        img = cv2.resize(img, TARGET_SIZE)
        img_dct = get_dct(img)
        img_dct = np.expand_dims(img_dct, axis=-1)
        img_dct = np.expand_dims(img_dct, axis=0)
        img_dct = img_dct / 255.0
        return img_dct
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # *** ADDED DEBUGGING ***
    print(f"\n--- Request Method: {request.method} ---")

    # Initialize variables at the start of the function scope
    prediction_result = None
    uploaded_filename_for_template = None
    mask_filename_for_template = None

    # *** ADDED DEBUGGING ***
    print(f"Initial state: prediction={prediction_result}, uploaded={uploaded_filename_for_template}, mask={mask_filename_for_template}")

    if request.method == 'POST':
        # *** ADDED DEBUGGING ***
        print("--- Inside POST block ---")

        # Check if model loaded
        if model is None:
             flash('Model not loaded. Cannot process request.', 'danger')
             # Go straight to rendering the template for GET-like state but show flash
             response = make_response(render_template('index.html',
                                        prediction=prediction_result,
                                        uploaded_image=uploaded_filename_for_template,
                                        mask_image=mask_filename_for_template,
                                        FORGERY_THRESHOLD=FORGERY_THRESHOLD))
             response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
             response.headers['Pragma'] = 'no-cache'
             response.headers['Expires'] = '0'
             return response

        # Check file upload
        if 'image' not in request.files:
            flash('No image file selected.', 'warning')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No selected file.', 'warning')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save Uploaded File
            base, ext = os.path.splitext(secure_filename(file.filename))
            unique_filename = f"{base}_{int(time.time())}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            try:
                file.save(filepath)
                print(f"File saved to {filepath}")
                uploaded_filename_for_template = f'uploads/{unique_filename}' # Use forward slash
            except Exception as e:
                print(f"Error saving uploaded file: {e}")
                flash(f"Error saving file: {e}", "danger")
                # Render template if save fails, variables might still be None
                response = make_response(render_template('index.html',
                                            prediction=prediction_result,
                                            uploaded_image=uploaded_filename_for_template,
                                            mask_image=mask_filename_for_template,
                                            FORGERY_THRESHOLD=FORGERY_THRESHOLD))
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'
                return response

            # Preprocess and Predict
            processed_image = preprocess_image(filepath)
            if processed_image is not None:
                try:
                    print("Running prediction...")
                    prediction = model.predict(processed_image)
                    print("Prediction complete.")

                    # Post-process
                    predicted_mask_prob = np.squeeze(prediction)
                    predicted_mask_binary = predicted_mask_prob.copy()
                    predicted_mask_binary[predicted_mask_binary > 0.5] = 1.0
                    predicted_mask_binary[predicted_mask_binary <= 0.5] = 0.0
                    forged_pixel_percentage = np.sum(predicted_mask_binary) / (TARGET_SIZE[0] * TARGET_SIZE[1])
                    classification = "FORGED" if forged_pixel_percentage > FORGERY_THRESHOLD else "REAL"

                    # Save the mask image
                    mask_image_to_save = (predicted_mask_binary * 255).astype(np.uint8)
                    mask_unique_filename = f"mask_{unique_filename}"
                    mask_filepath = os.path.join(app.config['MASK_FOLDER'], mask_unique_filename)
                    try:
                         cv2.imwrite(mask_filepath, mask_image_to_save)
                         print(f"Mask saved to {mask_filepath}")
                         mask_filename_for_template = f'masks/{mask_unique_filename}' # Use forward slash
                    except Exception as e:
                        print(f"Error saving mask image: {e}")
                        flash(f"Error saving mask image: {e}", "danger")

                    # Prepare results for template
                    prediction_result = {
                        "classification": classification,
                        "forged_percentage": f"{forged_pixel_percentage * 100:.2f}%"
                    }
                    flash('Prediction successful!', 'success')

                except Exception as e:
                    print(f"Error during prediction or post-processing: {e}")
                    flash(f'Error during prediction: {e}', 'danger')
            else:
                flash('Error preprocessing the image.', 'danger')

        else: # File not allowed type
            flash('Invalid file type. Allowed types: png, jpg, jpeg, bmp, tiff', 'warning')
            # Render template directly to show flash message
            response = make_response(render_template('index.html',
                                        prediction=prediction_result,
                                        uploaded_image=uploaded_filename_for_template,
                                        mask_image=mask_filename_for_template,
                                        FORGERY_THRESHOLD=FORGERY_THRESHOLD))
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response

    # --- Render Template (Reached for GET and end of POST) ---
    # *** ADDED DEBUGGING ***
    print(f"--- Preparing final response (Method: {request.method}) ---")
    print(f"DEBUG: Passing to template - prediction = {prediction_result}")
    print(f"DEBUG: Passing to template - uploaded_image = {uploaded_filename_for_template}")
    print(f"DEBUG: Passing to template - mask_image = {mask_filename_for_template}")

    response = make_response(render_template('index.html',
                           prediction=prediction_result,
                           uploaded_image=uploaded_filename_for_template,
                           mask_image=mask_filename_for_template,
                           FORGERY_THRESHOLD=FORGERY_THRESHOLD))
    # Add cache headers
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)