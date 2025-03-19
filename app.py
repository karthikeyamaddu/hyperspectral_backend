import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import matplotlib.pyplot as plt
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# Save uploads inside static/uploads so they are accessible by the frontend
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join('static', 'output')

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load your hyperspectral image classification model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def extract_patches(image, window_size):
    """
    Extract patches from the image using a sliding window.
    Only positions where a full window is available are processed.
    Returns:
        patches: numpy array of shape (n_patches, window_size, window_size, channels)
        positions: list of (i, j) tuples indicating the center pixel of each patch.
    """
    margin = window_size // 2
    H, W, C = image.shape
    patches = []
    positions = []
    
    for i in range(margin, H - margin):
        for j in range(margin, W - margin):
            patch = image[i - margin : i + margin + 1, j - margin : j + margin + 1, :]
            patches.append(patch)
            positions.append((i, j))
    
    patches = np.array(patches)
    return patches, positions

def reconstruct_prediction(positions, predictions, image_shape):
    """
    Reconstruct a full-size prediction image given the positions and predicted labels.
    The predictions array is assumed to be of length equal to the number of patches.
    Positions is a list of (i, j) tuples corresponding to each prediction.
    """
    H, W = image_shape[:2]
    pred_img = np.zeros((H, W), dtype=int)
    
    # Assign each predicted label (adding 1 if needed, to match expected label numbering)
    for (i, j), pred in zip(positions, predictions):
        pred_img[i, j] = pred + 1  
    return pred_img

def label_to_rgb(label_img):
    """
    Convert a label image (with integer class labels) to an RGB image.
    Adjust the color mapping as needed.
    """
    # Define color mapping: key = label, value = [R, G, B]
    color_mapping = {
        0: [0, 0, 0],        # Black for background/invalid
        1: [0, 0, 255],      # Blue
        2: [0, 255, 0],      # Green
        3: [255, 255, 0],    # Yellow
        4: [255, 165, 0],    # Orange
        5: [128, 0, 128],    # Purple
        6: [165, 42, 42],    # Brown
        7: [255, 192, 203],  # Pink
        8: [128, 128, 128],  # Gray
        9: [0, 255, 255]     # Cyan
    }
    
    H, W = label_img.shape
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in color_mapping.items():
        rgb_img[label_img == label] = color
    return rgb_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Validate file upload
    if 'image' not in request.files:
        return jsonify(message="No file part in request"), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify(message="No file selected"), 400
    
    # Save uploaded file to static/uploads for easy access from the frontend
    upload_filename = file.filename
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
    file.save(upload_path)
    
    try:
        # Open image using PIL and convert to RGB
        image = Image.open(upload_path)
        image = image.convert("RGB")
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # If the model expects 10 channels but the image has 3, tile the channels to create 10 channels.
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            reps = int(np.ceil(10 / 3))
            image_np = np.tile(image_np, (1, 1, reps))[:, :, :10]
        elif image_np.ndim == 2:
            # If grayscale, convert to 3 channels then tile
            image_np = np.stack([image_np]*3, axis=-1)
            reps = int(np.ceil(10 / 3))
            image_np = np.tile(image_np, (1, 1, reps))[:, :, :10]
        
        # Normalize image (scale to [0,1])
        image_np = image_np.astype('float32') / 255.0
        
        # Set the window size (patch spatial dimensions)
        window_size = 5
        
        # Ensure the image is large enough for patch extraction
        H, W, C = image_np.shape
        if H < window_size or W < window_size:
            return jsonify(message="Image is too small for the given window size."), 400
        
        # Extract patches from the image (using sliding window)
        patches, positions = extract_patches(image_np, window_size)
        
        # Use the model to predict class probabilities for each patch
        predictions_prob = model.predict(patches)
        # Convert probabilities to predicted class labels (using argmax)
        predictions = np.argmax(predictions_prob, axis=1)
        
        # Reconstruct a full prediction image from the patch predictions
        pred_label_img = reconstruct_prediction(positions, predictions, image_np.shape)
        
        # Convert the predicted label image to an RGB image for visualization
        rgb_output = label_to_rgb(pred_label_img)
        
        # Save the predicted RGB output image to static/output
        output_filename = 'classified_' + os.path.splitext(upload_filename)[0] + '.png'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        output_image = Image.fromarray(rgb_output)
        output_image.save(output_path)
        
        message = "Classification completed successfully."
    except Exception as e:
        message = f"Error during processing: {e}"
        output_filename = None
        upload_filename = None

    # Return a JSON response containing the filenames and message
    return jsonify(
        message=message,
        input_image=upload_filename,
        output_image=output_filename
    )

if __name__ == '__main__':
    app.run(debug=True)
