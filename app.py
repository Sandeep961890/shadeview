import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, send_from_directory # Added send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF
from datetime import datetime
import cv2  # For image processing
import numpy as np # For numerical operations with images

# --- NEW IMPORTS FOR AI (Machine Learning) INTEGRATION ---
from sklearn.neighbors import KNeighborsClassifier
import joblib # For saving/loading machine learning models
import pandas as pd # For CSV handling
# ---------------------------------------------------------

# --- REMOVED: import from image_processing.py (functions are now defined directly below) ---

# ===============================================
# 1. FLASK APPLICATION SETUP & CONFIGURATION
# ===============================================

app = Flask(__name__)
app.secret_key = "your_super_secret_and_long_random_key_here" # IMPORTANT: Change this in production!

# Define upload and report folders
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'

# Ensure these directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Configure Flask app with folder paths
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
app.config['DATABASE'] = os.path.join(app.root_path, 'database.db') # Database will be in the project root

# ===============================================
# 2. DATABASE INITIALIZATION & HELPERS
# ===============================================

def get_db():
    """Establishes a new database connection or returns the current one."""
    if 'db' not in g:
        g.db = sqlite3.connect(
            app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES,
            timeout=10 # Add a timeout to prevent database locked errors
        )
        g.db.row_factory = sqlite3.Row # Allows accessing columns by name
    return g.db

def close_db(e=None):
    """Closes the database connection at the end of the request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Register the close_db function to be called after each request
app.teardown_appcontext(close_db)

def init_db():
    """Initializes the database schema."""
    with app.app_context(): # Ensure we're in the app context to use get_db
        db = get_db()
        cursor = db.cursor()

        # Create users table
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            phone TEXT UNIQUE,
                            username TEXT,
                            password TEXT
                        )''')
        # Create patients table
        cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            op_number TEXT UNIQUE,
                            patient_name TEXT,
                            age INTEGER,
                            sex TEXT,
                            record_date TEXT,
                            FOREIGN KEY (user_id) REFERENCES users(id)
                        )''')
        db.commit()

# Call init_db to set up the database when the app starts
init_db()

# ===============================================
# 3. AUTHENTICATION HELPERS
# ===============================================

@app.before_request
def load_logged_in_user():
    """Loads the logged-in user into Flask's `g` object before each request."""
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        db = get_db()
        g.user = db.execute(
            'SELECT * FROM users WHERE id = ?', (user_id,)
        ).fetchone()

def login_required(view):
    """Decorator to protect routes that require a logged-in user."""
    import functools
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view

# ===============================================
# 4. CORE HELPER FUNCTIONS (Image Correction, Shade Detection, PDF Generation, New AI Placeholders)
# ===============================================

# --- IMAGE PROCESSING FUNCTIONS (Moved back into app.py) ---
def gray_world_white_balance(img):
    """
    Applies Gray World Algorithm for white balancing an image.
    Args:
        img (numpy.ndarray): The input image in BGR format.
    Returns:
        numpy.ndarray: The white-balanced image in BGR format.
    """
    result = img.copy().astype(np.float32) # Convert to float32 for calculations
    
    # Calculate average intensity for each channel
    avgB = np.mean(result[:, :, 0])
    avgG = np.mean(result[:, :, 1])
    avgR = np.mean(result[:, :, 2])
    
    # Calculate overall average gray value
    avgGray = (avgB + avgG + avgR) / 3

    # Apply scaling factor to each channel
    result[:, :, 0] = np.minimum(result[:, :, 0] * (avgGray / avgB), 255)
    result[:, :, 1] = np.minimum(result[:, :, 1] * (avgGray / avgG), 255)
    result[:, :, 2] = np.minimum(result[:, :, 2] * (avgGray / avgR), 255)

    return result.astype(np.uint8) # Convert back to uint8 for image display/saving


def correct_lighting(img_np_array):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting correction.
    Args:
        img_np_array (numpy.ndarray): The input image (NumPy array, BGR format).
    Returns:
        numpy.ndarray: The corrected image (NumPy array, BGR format).
    """
    # Convert to LAB color space
    img_lab = cv2.cvtColor(img_np_array, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(img_lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)

    # Merge channels back
    img_lab_eq = cv2.merge((l_eq, a, b))

    # Convert back to BGR color space
    img_corrected = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)
    return img_corrected
# --- END IMAGE PROCESSING FUNCTIONS ---


# Function to map L-value to a VITA shade based on adjusted rules
# This is a rule-based system for per-zone visual display, not the ML model
def map_l_to_shade_rule_based(l_value):
    # These thresholds are a heuristic and may need fine-tuning with more real data
    # B1 is typically the brightest VITA shade. A1 is slightly less bright, often more reddish.
    if l_value > 80: return "B1" # Very bright
    elif l_value > 75: return "A1" # Slightly less bright, typically A1 range
    elif l_value > 70: return "A2"
    elif l_value > 65: return "B2"
    elif l_value > 60: return "C2"
    elif l_value > 55: return "A3"
    else: return "C3" # Darker shades


# --- AI Model Setup (Loading Data from CSV & Training/Loading) ---
MODEL_FILENAME = "shade_classifier_model.pkl"
DATASET_FILENAME = "tooth_shades_simulated.csv"

def train_model():
    """Train a new KNN model using the CSV file and save it."""
    if not os.path.exists(DATASET_FILENAME):
        print(f"ERROR: Dataset '{DATASET_FILENAME}' is missing. Cannot train model.")
        return None # Return None if dataset is missing

    try:
        df = pd.read_csv(DATASET_FILENAME)
        if df.empty:
            print(f"ERROR: Dataset '{DATASET_FILENAME}' is empty. Cannot train model.")
            return None

        X = df[['incisal_l', 'middle_l', 'cervical_l']].values
        y = df['overall_shade'].values
        print(f"DEBUG: Training data shape={X.shape}, classes={np.unique(y)}")

        model_to_train = KNeighborsClassifier(n_neighbors=3)
        model_to_train.fit(X, y)
        joblib.dump(model_to_train, MODEL_FILENAME)
        print(f"DEBUG: Model trained and saved to {MODEL_FILENAME}")
        return model_to_train
    except Exception as e:
        print(f"ERROR: Failed to train model: {e}")
        return None

def load_or_train_model():
    """Load existing model or train a new one."""
    if os.path.exists(MODEL_FILENAME):
        try:
            loaded_model = joblib.load(MODEL_FILENAME)
            print(f"DEBUG: Loaded pre-trained shade model from {MODEL_FILENAME}")
            return loaded_model
        except Exception as e:
            print(f"WARNING: Could not load model from {MODEL_FILENAME}: {e}. Attempting to retrain.")
            return train_model()
    else:
        print(f"DEBUG: No existing model found at {MODEL_FILENAME}. Attempting to train new model.")
        return train_model()

# Global variable to store the loaded/trained model
# This will be initialized once when the app starts
shade_classifier_model = load_or_train_model()

# --- End AI Model Setup ---


# =========================================================
# NEW: Placeholder AI Modules for Advanced Analysis
# These functions would contain actual AI/ML model inference
# in a real-world application.
# =========================================================

def detect_face_features(image_np_array):
    """
    PLACEHOLDER: Simulates face detection and feature extraction.
    In a real application, this would use a pre-trained face detection
    model (e.g., MediaPipe Face Detection, OpenCV DNN with Haar cascades/DL models)
    to find facial landmarks and extract color/contrast properties.
    
    Args:
        image_np_array (numpy.ndarray): The input image (BGR format).
    Returns:
        dict: Simulated facial features.
    """
    print("DEBUG: Simulating Face Detection and Feature Extraction...")
    # Dummy data
    return {
        "skin_tone": "Warm (medium)", # e.g., "Cool", "Warm", "Neutral"
        "lip_color": "Pinkish-Red",   # e.g., "Pink", "Red", "Brown", "Purple"
        "eye_contrast": "High",       # e.g., "Low", "Medium", "High"
        "facial_harmony_score": 0.85  # A dummy score (0.0 to 1.0)
    }

def segment_and_analyze_teeth(image_np_array):
    """
    PLACEHOLDER: Simulates advanced tooth segmentation and shade analysis.
    In a real application, this would use:
    1. A tooth segmentation model (e.g., U-Net based CNN) to precisely
       identify and mask individual teeth.
    2. More sophisticated color analysis within the segmented regions
       (e.g., L*a*b* histograms, standard deviation, specific points).
    
    For now, it returns a simulated average LAB color for the whole tooth area,
    and then maps it to a VITA shade using an expanded set of rules.
    
    Args:
        image_np_array (numpy.ndarray): The pre-processed image (BGR format).
    Returns:
        dict: Simulated overall tooth shade characteristics.
    """
    print("DEBUG: Simulating Tooth Segmentation and Detailed Shade Analysis...")
    # For simulation, let's just convert the whole image to LAB and take average
    # In reality, this would be done on *segmented* tooth regions.
    img_lab = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2LAB)
    
    avg_l = np.mean(img_lab[:,:,0]) # Average Lightness (L channel is 0-100)
    avg_a = np.mean(img_lab[:,:,1]) # Average Green-Red component (a channel is -128 to 127)
    avg_b = np.mean(img_lab[:,:,2]) # Average Blue-Yellow component (b channel is -128 to 127)

    # Dummy mapping for overall shade based on average L and b values (expanded and very simplified)
    # This is *not* a real shade matching algorithm, just for simulation purposes.
    # The thresholds here are arbitrary for demonstration and would need proper
    # calibration with a real dataset of images and their known VITA shades.
    # This logic aims to cover A1, A2, B1, B2, C1, C2, C3 based on L and B values.
    if avg_l > 78:
        if avg_b > 18:
            simulated_overall_shade = "B1 (Simulated)"
        elif avg_b > 12:
            simulated_overall_shade = "A1 (Simulated)"
        else:
            simulated_overall_shade = "Uncategorized Bright (Simulated)" # Catch for variations
    elif avg_l > 73:
        if avg_b > 12:
            simulated_overall_shade = "A1 (Simulated)" # Can overlap with previous range based on L
        elif avg_b > 8:
            simulated_overall_shade = "B2 (Simulated)"
        else:
            simulated_overall_shade = "Uncategorized Medium (Simulated)"
    elif avg_l > 68:
        if avg_b > 4:
            simulated_overall_shade = "A2 (Simulated)"
        elif avg_b > 0:
            simulated_overall_shade = "C1 (Simulated)"
        else:
            simulated_overall_shade = "Uncategorized Darker (Simulated)"
    elif avg_l > 63:
        if avg_b > -5:
            simulated_overall_shade = "C2 (Simulated)"
        else:
            simulated_overall_shade = "Uncategorized Very Dark (Simulated)"
    elif avg_l > 58: # A3 is generally darker than A1/A2, often more reddish
        simulated_overall_shade = "A3 (Simulated)"
    elif avg_l > 50:
        simulated_overall_shade = "C3 (Simulated)" # Darkest C-shade
    else:
        simulated_overall_shade = "Uncategorized (Simulated)" # Fallback for very dark/unusual values

    return {
        "overall_lab": {"L": avg_l, "a": avg_a, "b": avg_b},
        "simulated_overall_shade": simulated_overall_shade,
        "tooth_condition": "Normal", # Placeholder: could be "Discolored", "Fluorosis", "Carious"
        "stain_presence": "None",    # Placeholder: "Light", "Moderate", "Heavy"
        "decay_presence": "None"     # Placeholder: "Yes", "No"
    }


def aesthetic_shade_suggestion(facial_features, tooth_analysis):
    """
    PLACEHOLDER: Simulates an aesthetic mapping model.
    In a real application, this would be an ML model trained on a dataset
    of facial aesthetics and desired tooth shades (e.g., from cosmetic dentistry cases).
    It would take facial features and current tooth shade characteristics
    to suggest an "ideal" cosmetic shade.
    
    Args:
        facial_features (dict): Output from detect_face_features.
        tooth_analysis (dict): Output from segment_and_analyze_teeth.
    Returns:
        dict: Simulated aesthetic recommendations.
    """
    print("DEBUG: Simulating Aesthetic Mapping and Shade Suggestion...")
    # Dummy logic for aesthetic suggestion based on simulated inputs
    suggested_shade = "N/A"
    confidence = "Low"
    
    if facial_features["eye_contrast"] == "High" and tooth_analysis["simulated_overall_shade"].startswith("A"):
        suggested_shade = "A1 (Aesthetic Suggestion)"
        confidence = "Medium"
    elif facial_features["skin_tone"] == "Warm (medium)" and tooth_analysis["simulated_overall_shade"].startswith("B"):
        suggested_shade = "B1 (Aesthetic Suggestion)"
        confidence = "Medium"
    elif tooth_analysis["simulated_overall_shade"].startswith("C"):
        suggested_shade = "A2 (Aesthetic Suggestion - Brightening)"
        confidence = "High" # Suggest brightening for darker C shades
    else:
        suggested_shade = "No specific aesthetic suggestion (Simulated)"
        confidence = "Very Low"

    return {
        "suggested_aesthetic_shade": suggested_shade,
        "aesthetic_confidence": confidence,
        "recommendation_notes": "This is a simulated aesthetic suggestion. Consult a specialist for detailed cosmetic planning."
    }

# =========================================================
# END NEW: Placeholder AI Modules
# =========================================================


def detect_shades_from_image(image_path):
    """
    Performs lighting correction, white balance, extracts features,
    and then uses the pre-trained ML model for overall tooth shade detection.
    Also, provides rule-based shades for individual zones for UI consistency.
    This now also calls new placeholder AI modules.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return {
                "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
                "overall_ml_shade": "N/A",
                "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {}
            }

        # --- Apply Image Pre-processing (functions are now local) ---
        # 1. Apply Gray World White Balance to correct color casts
        img_wb = gray_world_white_balance(img)
        
        # 2. Apply CLAHE for contrast and brightness normalization
        img_corrected = correct_lighting(img_wb)

        # --- NEW: Call placeholder AI modules ---
        # For full facial photo analysis
        face_features = detect_face_features(img_corrected) 
        
        # For detailed tooth segmentation and analysis
        tooth_analysis = segment_and_analyze_teeth(img_corrected)

        # For aesthetic shade suggestion
        aesthetic_suggestion = aesthetic_shade_suggestion(face_features, tooth_analysis)
        # ------------------------------------

        height, _, _ = img_corrected.shape # Use the corrected image for feature extraction
        
        # Define approximate regions for zones (top 30%, middle 40%, bottom 30%)
        # Note: In a real system, these zones would come from precise tooth segmentation
        incisal_zone = img_corrected[0:int(height*0.3), :, :]
        middle_zone = img_corrected[int(height*0.3):int(height*0.7), :, :]
        cervical_zone = img_corrected[int(height*0.7):height, :, :]

        # Convert zones to LAB color space for better perceptual uniformity
        incisal_lab = cv2.cvtColor(incisal_zone, cv2.COLOR_BGR2LAB)
        middle_lab = cv2.cvtColor(middle_zone, cv2.COLOR_BGR2LAB)
        cervical_lab = cv2.cvtColor(cervical_zone, cv2.COLOR_BGR2LAB)

        # Calculate average L (lightness) channel for each zone (L channel is 0-100)
        avg_incisal_l = np.mean(incisal_lab[:,:,0])
        avg_middle_l = np.mean(middle_lab[:,:,0])
        avg_cervical_l = np.mean(cervical_lab[:,:,0])

        # --- AI-BASED OVERALL SHADE PREDICTION (from the trained K-NN model) ---
        # IMPORTANT NOTE ON DISCREPANCIES:
        # The 'overall_ml_shade' is predicted by the K-NN model trained on 'tooth_shades_simulated.csv'.
        # The 'incisal', 'middle', 'cervical' shades are from the 'map_l_to_shade_rule_based' function,
        # which uses fixed L-value thresholds.
        # These two prediction methods use different logic and data, and therefore,
        # discrepancies (e.g., C3 per-zone vs B1 overall) can occur, especially with
        # limited and simulated training data for the K-NN model.
        # For real-world accuracy, a much larger, diverse, and expertly-annotated
        # dataset with advanced AI models would be required to ensure consistency
        # between regional and overall shade predictions.
        if shade_classifier_model is not None:
            features_for_ml_prediction = np.array([[avg_incisal_l, avg_middle_l, avg_cervical_l]])
            overall_ml_shade = shade_classifier_model.predict(features_for_ml_prediction)[0] 
        else:
            overall_ml_shade = "Model Error"
            print("WARNING: AI model not loaded/trained. Cannot provide ML shade prediction.")
        # ------------------------------------------

        # --- RULE-BASED PER-ZONE SHADE MAPPING (for detailed report display) ---
        # This function provides per-zone shade values for UI consistency.
        # It uses the map_l_to_shade_rule_based defined globally.
        detected_shades = {
            "incisal": map_l_to_shade_rule_based(avg_incisal_l),
            "middle": map_l_to_shade_rule_based(avg_middle_l),
            "cervical": map_l_to_shade_rule_based(avg_cervical_l),
            "overall_ml_shade": overall_ml_shade, # Include the AI's overall prediction from the KNN model
            # Pass the results from new AI modules
            "face_features": face_features,
            "tooth_analysis": tooth_analysis,
            "aesthetic_suggestion": aesthetic_suggestion
        }
        
        print(f"DEBUG: Features for ML: {features_for_ml_prediction}")
        print(f"DEBUG: Predicted Overall Shade (ML): {overall_ml_shade}")
        print(f"DEBUG: Detected Shades Per Zone (Rule-based): {detected_shades}")
        return detected_shades

    except Exception as e:
        print(f"Error during shade detection: {e}")
        return {
            "incisal": "Error", "middle": "Error", "cervical": "Error",
            "overall_ml_shade": "Error",
            "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {}
        }


def generate_pdf_report(patient_name, shades, image_path, filepath):
    """Generates a PDF report with detected shades and the uploaded image."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)

    pdf.cell(200, 10, txt="Shade View - Tooth Shade Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True) # Use patient_name
    pdf.cell(0, 10, txt=f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Detected Shades:", ln=True)
    pdf.set_font("Arial", size=12)
    # Display overall AI shade if available
    if "overall_ml_shade" in shades and shades["overall_ml_shade"] != "N/A":
        pdf.cell(0, 7, txt=f"   - Overall AI Prediction (K-NN): {shades['overall_ml_shade']}", ln=True)
    
    # Display per-zone shades (rule-based or from more advanced tooth analysis)
    pdf.cell(0, 7, txt=f"   - Incisal Zone: {shades['incisal']}", ln=True)
    pdf.cell(0, 7, txt=f"   - Middle Zone: {shades['middle']}", ln=True)
    pdf.cell(0, 7, txt=f"   - Cervical Zone: {shades['cervical']}", ln=True)
    
    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=13)
    pdf.cell(0, 10, txt="Advanced AI Insights (Simulated):", ln=True)
    pdf.set_font("Arial", size=11)

    # Display Simulated Tooth Analysis (e.g., overall LAB and condition)
    tooth_analysis = shades.get("tooth_analysis", {})
    if tooth_analysis:
        pdf.cell(0, 7, txt=f"  - Simulated Tooth Condition: {tooth_analysis.get('tooth_condition', 'N/A')}", ln=True)
        # Ensure values exist before formatting
        l_val = tooth_analysis.get('overall_lab', {}).get('L', 'N/A')
        a_val = tooth_analysis.get('overall_lab', {}).get('a', 'N/A')
        b_val = tooth_analysis.get('overall_lab', {}).get('b', 'N/A')
        
        # Check if values are numeric before trying to format
        if all(isinstance(v, (int, float)) for v in [l_val, a_val, b_val]):
            pdf.cell(0, 7, txt=f"  - Simulated Overall LAB: L={l_val:.2f}, a={a_val:.2f}, b={b_val:.2f}", ln=True)
        else:
            pdf.cell(0, 7, txt=f"  - Simulated Overall LAB: L={l_val}, a={a_val}, b={b_val}", ln=True)
        
        pdf.cell(0, 7, txt=f"  - Simulated Overall Shade (Advanced): {tooth_analysis.get('simulated_overall_shade', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"  - Simulated Stain Presence: {tooth_analysis.get('stain_presence', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"  - Simulated Decay Presence: {tooth_analysis.get('decay_presence', 'N/A')}", ln=True)


    pdf.ln(5)
    # Display Simulated Facial Features
    face_features = shades.get("face_features", {})
    if face_features:
        pdf.cell(0, 7, txt=f"  - Simulated Facial Skin Tone: {face_features.get('skin_tone', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"  - Simulated Lip Color: {face_features.get('lip_color', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"  - Simulated Eye Contrast: {face_features.get('eye_contrast', 'N/A')}", ln=True)
        
        harmony_score = face_features.get('facial_harmony_score', 'N/A')
        if isinstance(harmony_score, (int, float)):
            pdf.cell(0, 7, txt=f"  - Simulated Facial Harmony Score: {harmony_score:.2f}", ln=True)
        else:
            pdf.cell(0, 7, txt=f"  - Simulated Facial Harmony Score: {harmony_score}", ln=True)


    pdf.ln(5)
    # Display Simulated Aesthetic Suggestion
    aesthetic_suggestion = shades.get("aesthetic_suggestion", {})
    if aesthetic_suggestion:
        pdf.cell(0, 7, txt=f"  - Suggested Aesthetic Shade: {aesthetic_suggestion.get('suggested_aesthetic_shade', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"  - Aesthetic Confidence: {aesthetic_suggestion.get('aesthetic_confidence', 'N/A')}", ln=True)
        pdf.multi_cell(0, 7, txt=f"  - Recommendation Notes: {aesthetic_suggestion.get('recommendation_notes', 'N/A')}")
    
    pdf.ln(10)

    try:
        if os.path.exists(image_path):
            pdf.cell(0, 10, txt="Uploaded Image:", ln=True)
            if pdf.get_y() > 200:
                pdf.add_page()
            # Calculate aspect ratio to fit image on page
            img_cv = cv2.imread(image_path)
            if img_cv is not None:
                h_img, w_img, _ = img_cv.shape
                # Max width for image on PDF page
                max_w_pdf = 180 # A4 width approx 210, with 10mm margins on each side
                # Calculate height based on aspect ratio
                w_pdf = min(w_img, max_w_pdf)
                h_pdf = h_img * (w_pdf / w_img)
                
                # If image too tall for remaining space, put on new page
                if pdf.get_y() + h_pdf + 10 > pdf.h - pdf.b_margin: # Check if image fits with some buffer
                     pdf.add_page()
                     
                pdf.image(image_path, x=pdf.get_x(), y=pdf.get_y(), w=w_pdf, h=h_pdf)
                pdf.ln(h_pdf + 10) # Move cursor down after image
            else:
                 pdf.cell(0, 10, txt="Note: Image could not be loaded for embedding.", ln=True)

        else:
            pdf.cell(0, 10, txt="Note: Uploaded image file not found for embedding.", ln=True)
    except Exception as e:
        print(f"Error adding image to PDF: {e}")
        pdf.cell(0, 10, txt="Note: An error occurred while embedding the image in the report.", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt="This report provides an estimated tooth shade based on automated analysis and simulated advanced AI.", ln=True)
    pdf.cell(0, 10, txt="Please consult with a dental professional for definitive assessment, diagnosis, and treatment planning.", ln=True)

    pdf.output(filepath)


# ===============================================
# 5. ROUTES
# ===============================================

@app.route('/')
def home():
    """Renders the home/landing page."""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if request.method == 'POST':
        phone = request.form['phone']
        password = request.form['password']
        db = get_db()
        error = None

        user = db.execute(
            'SELECT * FROM users WHERE phone = ?', (phone,)
        ).fetchone()

        if user is None:
            error = 'Incorrect phone number.'
        elif not check_password_hash(user['password'], password):
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard')) # Redirect to the dashboard (Patient Entry)
        
        flash(error, 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles new user registration."""
    if request.method == 'POST':
        phone = request.form['phone']
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None

        if not all([phone, username, password]):
            error = 'All fields are required.'

        if error is None:
            try:
                db.execute(
                    "INSERT INTO users (phone, username, password) VALUES (?, ?, ?)",
                    (phone, username, generate_password_hash(password)),
                )
                db.commit()
            except sqlite3.IntegrityError:
                error = f"Phone number {phone} is already registered."
            except Exception as e:
                error = f"Registration failed: {e}"
            else:
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for("login"))
        
        flash(error, 'error')
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logs out the current user."""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard') # This is now the Patient Entry Page
@login_required
def dashboard():
    """Renders the patient entry form and a list of recently added patients."""
    db = get_db()
    # Fetch recent patients for the logged-in user to display
    patients = db.execute(
        "SELECT op_number, patient_name, age, sex, record_date FROM patients WHERE user_id = ? ORDER BY record_date DESC LIMIT 5",
        (g.user['id'],)
    ).fetchall()
    return render_template('dashboard.html', username=g.user['username'], patients=patients)


@app.route('/save_patient_data', methods=['POST'])
@login_required
def save_patient_data():
    """Handles saving new patient records and redirects to image upload page."""
    op_number = request.form['op_number']
    patient_name = request.form['patient_name']
    age = request.form['age']
    sex = request.form['sex']
    record_date = request.form['date']
    user_id = g.user['id']
    db = get_db()
    error = None

    if not all([op_number, patient_name, age, sex, record_date]):
        error = "All patient fields are required."

    if error is None:
        try:
            db.execute(
                "INSERT INTO patients (user_id, op_number, patient_name, age, sex, record_date) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, op_number, patient_name, age, sex, record_date)
            )
            db.commit()
            flash('Patient record saved successfully! Now upload an image.', 'success')
            # Redirect to the image upload page, passing the unique OP Number
            return redirect(url_for('upload_page', op_number=op_number))
        except sqlite3.IntegrityError:
            error = 'OP Number already exists for another patient. Please use a unique OP Number or select from recent entries.'
        except Exception as e:
            error = f'Error saving patient record: {e}'

    if error:
        flash(error, 'error')
    return redirect(url_for('dashboard')) # Go back to dashboard on error

@app.route('/upload_page/<op_number>')
@login_required
def upload_page(op_number):
    """Renders the dedicated image upload page for a specific patient."""
    db = get_db()
    patient = db.execute(
        "SELECT patient_name FROM patients WHERE op_number = ? AND user_id = ?",
        (op_number, g.user['id'])
    ).fetchone()

    if patient is None:
        flash('Patient not found or unauthorized access.', 'error')
        return redirect(url_for('dashboard'))
        
    return render_template('upload_page.html', op_number=op_number, patient_name=patient['patient_name'])


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    """Handles image upload, shade detection, and PDF report generation."""
    op_number = request.form.get('op_number') # Get op_number from hidden input

    if not op_number:
        flash('Patient OP Number missing for upload.', 'error')
        return redirect(url_for('dashboard')) # Fallback to dashboard

    db = get_db()
    patient = db.execute(
        "SELECT patient_name FROM patients WHERE op_number = ? AND user_id = ?",
        (op_number, g.user['id'])
    ).fetchone()

    if patient is None:
        flash('Patient not found or unauthorized to upload for this patient.', 'error')
        return redirect(url_for('dashboard')) # Fallback

    if 'image' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(url_for('upload_page', op_number=op_number))

    file = request.files['image']
    if file.filename == '':
        flash('No selected file.', 'error')
        return redirect(url_for('upload_page', op_number=op_number))

    if file:
        filename = secure_filename(file.filename)
        # Append OP number to filename to ensure uniqueness and link
        base, ext = os.path.splitext(filename)
        unique_filename = f"{op_number}_{base}{ext}" 
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # --- Perform Image Pre-processing and Shade Analysis (now includes ML) ---
        detected_shades = detect_shades_from_image(filepath) # This now includes new AI module outputs

        # Generate PDF Report
        report_filename = f"{op_number}_report_{os.path.splitext(unique_filename)[0]}.pdf"
        report_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)
        
        generate_pdf_report(patient['patient_name'], detected_shades, filepath, report_path)

        flash(f'Image uploaded and analysis complete for {patient["patient_name"]}. Report generated!', 'success')
        
        # Pass ALL detected shades (including new AI outputs) to the template
        return render_template(
            'report.html',
            user_name=patient['patient_name'],
            shades=detected_shades,
            report_filename=report_filename,
            # Explicitly pass the top-level keys for easy access in HTML if preferred
            overall_ml_shade=detected_shades.get('overall_ml_shade', 'N/A'),
            face_features=detected_shades.get('face_features', {}),
            tooth_analysis=detected_shades.get('tooth_analysis', {}),
            aesthetic_suggestion=detected_shades.get('aesthetic_suggestion', {})
        )

    flash('File upload failed.', 'error')
    return redirect(url_for('upload_page', op_number=op_number))

@app.route('/reports/<filename>')
@login_required
def download_report(filename):
    """Allows downloading generated PDF reports."""
    # Prevent directory traversal attacks
    if ".." in filename or filename.startswith("/"):
        flash("Invalid filename.", 'error')
        return redirect(url_for('dashboard'))

    # Optional: Add a check if the user is authorized to download this specific report
    # This would involve parsing the filename for the op_number or patient_id
    # and checking if it belongs to the logged-in user's patients.
    
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

# ===============================================
# 6. RUN THE APPLICATION
# ===============================================

if __name__ == '__main__':
    app.run(debug=True)
