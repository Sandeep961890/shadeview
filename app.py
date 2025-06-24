import os
import sqlite3  # Still imported but will not be actively used for new data handling
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF
from datetime import datetime
import cv2  # For image processing
import numpy as np  # For numerical operations with images
import json  # For parsing Firebase config
import traceback # Import for printing full tracebacks
import uuid # For generating unique filenames

# --- Firestore Imports ---
# These are conceptual imports for a standard Python environment.
# In the Canvas environment, Firebase interaction happens via fetch to API.
# However, we keep the imports for conceptual structure for a real Python Flask app.
# For this Canvas, we will simulate the Firestore calls.
# from firebase_admin import credentials, firestore, initialize_app
# from google.cloud.firestore import Client as FirestoreClient # For type hinting if using client library

# --- NEW IMPORTS FOR AI (Machine Learning) INTEGRATION ---
from sklearn.neighbors import KNeighborsClassifier
import joblib  # For saving/loading machine learning models
import pandas as pd  # For CSV handling
# ---------------------------------------------------------

# --- IMAGE PROCESSING FUNCTIONS (Self-contained for simplicity) ---
def gray_world_white_balance(img):
    """
    Applies Gray World Algorithm for white balancing an image.
    Args:
        img (numpy.ndarray): The input image in BGR format.
    Returns:
        numpy.ndarray: The white-balanced image in BGR format.
    """
    result = img.copy().astype(np.float32)  # Convert to float32 for calculations

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
    return result.astype(np.uint8)  # Convert back to uint8 for image display/saving


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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    # Merge channels back
    img_lab_eq = cv2.merge((l_eq, a, b))

    # Convert back to BGR color space
    img_corrected = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)
    return img_corrected


# --- END IMAGE PROCESSING FUNCTIONS ---


# ===============================================
# 1. FLASK APPLICATION SETUP & CONFIGURATION
# ===============================================

app = Flask(__name__)
# Secret key from environment variable for production readiness
app.secret_key = os.environ.get("SECRET_KEY", "your_strong_dev_secret_key_12345")

# Define upload and report folders
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'

# Ensure these directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Configure Flask app with folder paths
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
# app.config['DATABASE'] = os.path.join(app.root_path, 'database.db') # SQLite DB path (no longer primary)

# --- Firestore (Simulated for Canvas) ---
# In a real Flask app, you'd use Firebase Admin SDK or Google Cloud client library.
# Here, we'll use placeholder functions that simulate Firestore behavior
# using the Canvas provided globals.

# __app_id and __firebase_config are provided by the Canvas environment.
app_id = os.environ.get('__app_id', 'default-app-id')  # Fallback for local testing
firebase_config_str = os.environ.get('__firebase_config', '{}')
firebase_config = json.loads(firebase_config_str)

# Global for simulated Firestore database object
# This will be initialized once as a simple dict to hold all simulated data
db_data = {
    'artifacts': {
        app_id: {
            'users': {},
            'public': {'data': {}}
        }
    }
}
# 'db' will now just be a reference to this global dict for functions.
db = db_data # db is now a reference to the global simulated data structure

# --- Firestore Initialization Logic (moved to a function not directly called by app_context) ---
# This function is now mainly for conceptual understanding or if you have specific
# setup logic that needs to run only once *outside* the request lifecycle but *after* app init.
# For user session management, `before_request` is better.
def setup_initial_firebase_globals():
    """
    Sets up conceptual global data for Firestore simulation if needed.
    This runs once at app startup.
    """
    # This function mostly ensures app_id and firebase_config are processed.
    # User-specific G objects are handled in before_request.
    print(f"DEBUG: App ID: {app_id}")
    print(f"DEBUG: Firebase Config (partial): {list(firebase_config.keys())[:3]}...")

# Call this once at the module level or on app init if needed
setup_initial_firebase_globals()


# ===============================================
# 2. DATABASE INITIALIZATION & HELPERS (Firestore)
# ===============================================

# These functions now work with the simulated Firestore 'db_data' global

def get_firestore_collection(path_segments):
    """Navigates the simulated Firestore structure to get a collection."""
    current_level = db_data # Use the global db_data directly
    for segment in path_segments:
        if segment not in current_level:
            current_level[segment] = {}
        current_level = current_level[segment]
    return current_level


def get_firestore_document(path_segments):
    """Navigates the simulated Firestore structure to get a document."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    return collection.get(doc_id)


def set_firestore_document(path_segments, data):
    """Sets a document in the simulated Firestore."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore set: {os.path.join(*path_segments)}")


def add_firestore_document(path_segments, data):
    """Adds a document with auto-generated ID in the simulated Firestore."""
    collection = get_firestore_collection(path_segments)
    doc_id = str(np.random.randint(100000, 999999))  # Simulate auto-ID
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore added: {os.path.join(*path_segments)}/{doc_id}")
    return doc_id  # Return the simulated ID


def get_firestore_documents_in_collection(path_segments, query_filters=None):
    """Gets documents from a simulated Firestore collection, with basic filtering."""
    collection = get_firestore_collection(path_segments)
    results = []
    for doc_id, doc_data in collection.items():
        if query_filters:
            match = True
            for field, value in query_filters.items():
                if doc_data.get(field) != value:
                    match = False
                    break
            if match:
                results.append(doc_data)
        else:
            results.append(doc_data)

    # Simulate ordering by record_date if present
    if results and 'record_date' in results[0]:
        results.sort(key=lambda x: x.get('record_date', ''), reverse=True)
    return results


# No close_db for Firestore, as it's typically managed differently than SQLite.
# The db_data structure is persistent for the Flask app lifecycle.

# No init_db for SQLite needed, as Firestore is primary.
# The user/patient tables will now be collections in Firestore.

# ===============================================
# 3. AUTHENTICATION HELPERS (Adapted for Firestore)
# ===============================================

@app.before_request
def load_logged_in_user():
    """Loads the logged-in user into Flask's `g` object for the current request.
    Uses session for persistence across requests.
    """
    # If user_id is not in session, it's either a new session or the first request
    # Set up an anonymous user by default.
    if 'user_id' not in session:
        # In a Canvas environment, check for initial_auth_token.
        # Otherwise, assign a new anonymous ID.
        initial_auth_token = os.environ.get('__initial_auth_token')
        if initial_auth_token:
            session['user_id'] = initial_auth_token.split(':')[-1]
            session['user'] = {'id': session['user_id'], 'username': f"User_{session['user_id'][:8]}"}
            print(f"DEBUG: Initializing session user from token: {session['user']['username']}")
        else:
            session['user_id'] = 'anonymous-' + str(np.random.randint(100000, 999999))
            session['user'] = {'id': session['user_id'], 'username': f"AnonUser_{session['user_id'][-6:]}"}
            print(f"DEBUG: Initializing session user to anonymous: {session['user']['username']}")

    # Always assign from session to g for the current request
    g.user_id = session.get('user_id')
    g.user = session.get('user')

    # Ensure firestore_user_id is always available for Firestore pathing
    g.firestore_user_id = g.user_id # No need for a fallback here, as g.user_id is now guaranteed.

    # print(f"DEBUG: Request user loaded: {g.user['username']} (FS ID: {g.firestore_user_id})")


def login_required(view):
    """Decorator to protect routes that require a logged-in user (not anonymous)."""
    import functools

    @functools.wraps(view)
    def wrapped_view(**kwargs):
        # Check if the user is truly "logged in" (not the default anonymous user)
        # You might adjust this based on how real login works in the Canvas.
        if g.user is None or 'anonymous' in g.user_id: # Check if user is anonymous
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return view(**kwargs)

    return wrapped_view


# ===============================================
# 4. CORE HELPER FUNCTIONS (Image Correction, Shade Detection, PDF Generation, Enhanced Simulated AI)
# ===============================================

# Function to map L-value to a VITA shade based on adjusted rules
# This function will now expect L_value to be in 0-100 range
def map_l_to_shade_rule_based(l_value_100_scale): # Parameter name changed for clarity
    """
    Maps L-value (0-100 scale) to a VITA shade.
    Adjusted thresholds for a more granular mapping.
    """
    if l_value_100_scale > 85: # Very bright, whiter than B1
        return "B1+" # Custom for very bright
    elif l_value_100_scale > 80:
        return "B1"
    elif l_value_100_scale > 75:
        return "A1"
    elif l_value_100_scale > 70:
        return "B2"
    elif l_value_100_scale > 65:
        return "A2"
    elif l_value_100_scale > 60:
        return "D2" # Introduce D shades
    elif l_value_100_scale > 55:
        return "C1"
    elif l_value_100_scale > 50:
        return "A3"
    elif l_value_100_scale > 45:
        return "D3"
    elif l_value_100_scale > 40:
        return "B3"
    elif l_value_100_scale > 35:
        return "C2"
    elif l_value_100_scale > 30:
        return "A3.5"
    elif l_value_100_scale > 25:
        return "D4"
    elif l_value_100_scale > 20:
        return "C3"
    else:
        return "C4" # Darkest shades


# --- AI Model Setup (Loading Data from CSV & Training/Loading) ---
MODEL_FILENAME = "shade_classifier_model.pkl"
DATASET_FILENAME = "tooth_shades_simulated.csv"


def train_model():
    """Train a new KNN model using the CSV file and save it."""
    if not os.path.exists(DATASET_FILENAME):
        print(f"ERROR: Dataset '{DATASET_FILENAME}' is missing. Cannot train model.")
        return None

    try:
        df = pd.read_csv(DATASET_FILENAME)
        if df.empty:
            print(f"ERROR: Dataset '{DATASET_FILENAME}' is empty. Cannot train model.")
            return None

        # Ensure that the training data itself is expected to be in 0-100 scale for L values.
        # If your CSV's incisal_l, middle_l, cervical_l are already 0-255, this might need adjustment.
        # Based on previous context, these CSV values are assumed to be 0-100 L scale.
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
shade_classifier_model = load_or_train_model()

# --- End AI Model Setup ---


# =========================================================
# ENHANCED: Placeholder AI Modules for Advanced Analysis
# These functions now return more "intelligent" simulated data
# based on image properties, mimicking commercial insights.
# =========================================================

def detect_face_features(image_np_array):
    """
    ENHANCED PLACEHOLDER: Simulates detailed face feature extraction.
    Now attempts to derive more nuanced skin tone (including undertones),
    detailed lip color, and eye contrast based on average color properties
    and simple statistical analysis of the input image.

    Args:
        image_np_array (numpy.ndarray): The input image (BGR format).
    Returns:
        dict: Simulated facial features including skin and lip tone.
    """
    print("DEBUG: Simulating detailed Face Detection and Feature Extraction with color analysis...")

    img_lab = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2LAB)
    avg_l = np.mean(img_lab[:, :, 0])
    avg_a = np.mean(img_lab[:, :, 1])
    avg_b = np.mean(img_lab[:, :, 2])

    # Simulate Skin Tone and Undertone based on 'a' (green-red) and 'b' (blue-yellow) channels
    # More detailed heuristics for simulation
    skin_tone_category = "Medium"
    skin_undertone = "Neutral"

    if avg_l > 75:
        skin_tone_category = "Light"
    elif avg_l > 60:
        skin_tone_category = "Medium"
    elif avg_l > 45:
        skin_tone_category = "Dark"
    else:
        skin_tone_category = "Very Dark"

    if avg_b > 15 and avg_a > 8:
        skin_undertone = "Warm (Golden/Peach)"
    elif avg_b < 0 and avg_a < 5:
        skin_undertone = "Cool (Pink/Blue)"
    elif avg_b >= 0 and avg_a >= 5 and avg_a <= 8 and avg_b <= 15:
        skin_undertone = "Neutral"
    elif avg_b > 5 and avg_a < 5:
        skin_undertone = "Olive (Greenish)"

    simulated_skin_tone = f"{skin_tone_category} with {skin_undertone} undertones"

    # Simulate Lip Color based on 'a' (redness) and 'b' (yellowness) and L (lightness)
    simulated_lip_color = "Natural Pink"
    if avg_a > 20 and avg_l < 60:
        simulated_lip_color = "Deep Rosy Red"
    elif avg_a > 15 and avg_l >= 60:
        simulated_lip_color = "Bright Coral"
    elif avg_b < 5 and avg_l < 50:
        simulated_lip_color = "Subtle Mauve/Berry"
    elif avg_l < 50 and avg_a < 10 and avg_b < 10:
        simulated_lip_color = "Muted Nude/Brown"
    elif avg_l > 70 and avg_a < 10:
        simulated_lip_color = "Pale Nude"

    # Simulate Eye Contrast (using percentile spread in L-channel)
    l_channel = img_lab[:, :, 0]
    p10, p90 = np.percentile(l_channel, [10, 90])
    contrast_spread = p90 - p10
    eye_contrast_sim = "Medium"
    if contrast_spread > 40:
        eye_contrast_sim = "High (Distinct Features)"
    elif contrast_spread < 20:
        eye_contrast_sim = "Low (Soft Features)"

    return {
        "skin_tone": simulated_skin_tone,
        "lip_color": simulated_lip_color,
        "eye_contrast": eye_contrast_sim,
        "facial_harmony_score": round(np.random.uniform(0.7, 0.95), 2),  # FIX: Use round() for float
        # "notes": "Simulated analysis. Actual facial feature detection requires specialized deep learning models." # REMOVED
    }


def segment_and_analyze_teeth(image_np_array):
    """
    ENHANCED PLACEHOLDER: Simulates advanced tooth segmentation and shade analysis.
    Provides more detailed simulated insights on tooth condition and stain presence.

    Args:
        image_np_array (numpy.ndarray): The pre-processed image (BGR format).
    Returns:
        dict: Simulated overall tooth shade characteristics and condition.
    """
    print("DEBUG: Simulating detailed Tooth Segmentation and Analysis...")

    img_lab = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2LAB)

    avg_l = np.mean(img_lab[:, :, 0])
    avg_a = np.mean(img_lab[:, :, 1])
    avg_b = np.mean(img_lab[:, :, 2])

    # Simulate Overall Shade (expanded VITA-like mapping)
    if avg_l > 78:
        simulated_overall_shade = "B1 (High Brightness)"
    elif avg_l > 73:
        simulated_overall_shade = "A1 (Natural Brightness)"
    elif avg_l > 68:
        simulated_overall_shade = "A2 (Medium Brightness)"
    elif avg_l > 63:
        simulated_overall_shade = "B2 (Slightly Darker)"
    elif avg_l > 58:
        simulated_overall_shade = "C1 (Moderate Darkness)"
    elif avg_l > 53:
        simulated_overall_shade = "C2 (Noticeable Darkness)"
    elif avg_l > 48:
        simulated_overall_shade = "A3 (Darker, Reddish Tint)"
    else:
        simulated_overall_shade = "C3 (Very Dark)"

    # Simulate Tooth Condition based on 'L' and 'b' (yellowish tint)
    tooth_condition_sim = "Normal & Healthy Appearance"
    if avg_b > 20 and avg_l < 70:
        tooth_condition_sim = "Mild Discoloration (Yellowish)"
    elif avg_b > 25 and avg_l < 60:
        tooth_condition_sim = "Moderate Discoloration (Strong Yellow)"
    elif avg_l < 55 and avg_a > 10:
        tooth_condition_sim = "Pronounced Discoloration (Brown/Red)"
    elif avg_l < 60 and avg_b < 0:
        tooth_condition_sim = "Greyish Appearance"

    # Simulate Stain Presence based on contrast within tooth area (requires segmentation in real model)
    # For simulation, use a general image contrast heuristic
    l_std_dev = np.std(img_lab[:, :, 0])
    stain_presence_sim = "None detected"
    if l_std_dev > 25 and avg_l > 60:  # High contrast in brighter teeth could suggest stains
        stain_presence_sim = "Possible light surface stains"
    elif l_std_dev > 35 and avg_l < 60:  # High contrast in darker teeth
        stain_presence_sim = "Moderate localized stains"

    decay_presence_sim = "No visible signs of decay"  # Dummy
    if np.random.rand() < 0.05:  # 5% chance of simulating decay (random)
        decay_presence_sim = "Potential small carious lesion (simulated - consult professional)"

    return {
        "overall_lab": {"L": float(avg_l), "a": float(avg_a), "b": float(avg_b)},  # Ensure float for JSON
        "simulated_overall_shade": simulated_overall_shade,
        "tooth_condition": tooth_condition_sim,
        "stain_presence": stain_presence_sim,
        "decay_presence": decay_presence_sim,
        # "notes": "Simulated tooth analysis. Real analysis requires intra-oral images and advanced dental AI." # REMOVED
    }


def aesthetic_shade_suggestion(facial_features, tooth_analysis):
    """
    ENHANCED PLACEHOLDER: Simulates an aesthetic mapping model with more context.
    Suggestions are now more specific, considering simulated skin/lip tones.
    Confidence is now more dynamic based on harmony score and conditions.

    Args:
        facial_features (dict): Output from detect_face_features.
        tooth_analysis (dict): Output from segment_and_analyze_teeth.
    Returns:
        dict: Simulated aesthetic recommendations.
    """
    print("DEBUG: Simulating detailed Aesthetic Mapping and Shade Suggestion...")

    suggested_shade = "No specific aesthetic suggestion (Simulated)"
    aesthetic_confidence = "Low" # Changed default to 'Low'
    recommendation_notes = "This is a simulated aesthetic suggestion. Consult a dental specialist for personalized cosmetic planning based on your unique facial features and desired outcome. Advanced AI for aesthetics is complex and evolving."

    current_shade = tooth_analysis.get('simulated_overall_shade', '')
    skin_tone = facial_features.get('skin_tone', '').lower()
    lip_color = facial_features.get('lip_color', '').lower()
    facial_harmony_score = facial_features.get('facial_harmony_score', 0.5) # Get score, default to 0.5

    # Example sophisticated simulated logic with more dynamic confidence
    if "warm" in skin_tone:
        if "b1" in current_shade or "a1" in current_shade:
            suggested_shade = "Optimal Match (Simulated - Warm Undertone)"
            aesthetic_confidence = "Very High"
            recommendation_notes = "Your simulated warm skin undertone harmonizes exceptionally well with this bright shade, suggesting an optimal match. Consider maintaining this shade."
        elif "c3" in current_shade or "c2" in current_shade or "a3" in current_shade:
            suggested_shade = "B1 or A2 (Simulated - Brightening for Warm Undertone)"
            aesthetic_confidence = "High"
            recommendation_notes = "Your simulated warm skin undertone would be beautifully complemented by a brighter, slightly warmer tooth shade like B1 or A2. Consider professional whitening for a more radiant smile."
        else:
            aesthetic_confidence = "Medium" # Default for warm skin if no specific shade match

    elif "cool" in skin_tone:
        if "a1" in current_shade or "b1" in current_shade:
            suggested_shade = "Optimal Match (Simulated - Cool Undertone)"
            aesthetic_confidence = "Very High"
            recommendation_notes = "This shade provides excellent contrast and harmony with your simulated cool skin undertone, suggesting an optimal match. A very crisp and bright appearance."
        elif "a3" in current_shade or "b2" in current_shade or "d" in current_shade:
            suggested_shade = "A1 or B1 (Simulated - Brightening for Cool Undertone)"
            aesthetic_confidence = "High"
            recommendation_notes = "With your simulated cool skin undertone, a crisp, bright shade like A1 or B1 could enhance your overall facial harmony. Avoid overly yellow shades for best results."
        else:
            aesthetic_confidence = "Medium" # Default for cool skin if no specific shade match

    elif "neutral" in skin_tone:
        if "b1" in current_shade or "a1" in current_shade or "a2" in current_shade:
            suggested_shade = "Balanced Brightness (Simulated)"
            aesthetic_confidence = "High"
            recommendation_notes = "Your neutral skin tone offers great versatility. This shade provides a balanced and natural bright smile. Options for further brightening or warmth can be explored."
        else:
            aesthetic_confidence = "Medium" # Default for neutral skin

    elif "olive" in skin_tone:
        if "a2" in current_shade or "b2" in current_shade:
            suggested_shade = "Enhanced Natural (Simulated - Olive Tone)"
            aesthetic_confidence = "Medium"
            recommendation_notes = "For a simulated olive skin tone, a balanced brightening to A2 can provide a natural yet enhanced smile. Be mindful of shades that pull too much yellow or grey."
        elif "a1" in current_shade or "b1" in current_shade:
            suggested_shade = "Significant Brightening (Simulated - Olive Tone)"
            aesthetic_confidence = "High"
            recommendation_notes = "While your current shade provides a natural look, shades like A1 or B1 could offer a more noticeable brightening effect while maintaining harmony with your olive tone."
        else:
            aesthetic_confidence = "Low" # Default for olive skin if no specific shade match

    # General confidence boost based on Facial Harmony Score
    if facial_harmony_score >= 0.90:
        if aesthetic_confidence == "Low": # Upgrade 'Low' to 'Medium'
            aesthetic_confidence = "Medium"
        elif aesthetic_confidence == "Medium": # Upgrade 'Medium' to 'High'
            aesthetic_confidence = "High"
    elif facial_harmony_score >= 0.80 and aesthetic_confidence == "Low":
        aesthetic_confidence = "Medium" # Slightly boost low confidence if harmony is decent

    # Ensure notes are concise if confidence is high, or revert to general if still low
    if aesthetic_confidence == "Very High":
        # Recommendation notes are already specific for 'Optimal Match'
        pass
    elif aesthetic_confidence == "High":
        # Recommendation notes are already specific for 'Brightening'
        pass
    elif aesthetic_confidence == "Medium":
        if "Balanced Brightness" not in suggested_shade and "Enhanced Natural" not in suggested_shade:
            recommendation_notes = "This shade offers a natural and pleasing appearance. For more significant changes, a dental consultation is recommended."
    else: # Low or Very Low confidence
        suggested_shade = "Consult Dental Specialist (Simulated)"
        recommendation_notes = "Based on the simulated analysis, a personalized consultation with a dental specialist is highly recommended for tailored cosmetic planning due to the complexity of aesthetic matching."


    return {
        "suggested_aesthetic_shade": suggested_shade,
        "aesthetic_confidence": aesthetic_confidence,
        "recommendation_notes": recommendation_notes
    }


# =========================================================
# END ENHANCED: Placeholder AI Modules
# =========================================================


def detect_shades_from_image(image_path):
    """
    Performs lighting correction, white balance, extracts features,
    and then uses the pre-trained ML model for overall tooth shade detection.
    Also, provides rule-based shades for individual zones for UI consistency.
    This now also calls new placeholder AI modules.
    """
    print(f"DEBUG: Starting image processing for {image_path}")
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: cv2.imread returned None for image at {image_path}. File might be missing, corrupted, or not an image (e.g., non-image file type, invalid path).")
            return {
                "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
                "overall_ml_shade": "N/A",
                "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {}
            }
        print(f"DEBUG: Image loaded successfully. Shape: {img.shape}, Type: {img.dtype}")

        # Ensure the image is not empty after loading
        if img.size == 0:
            print(f"ERROR: Image loaded but is empty (0 size) for {image_path}.")
            return {
                "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
                "overall_ml_shade": "N/A",
                "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {}
            }


        # --- Apply Image Pre-processing ---
        img_wb = gray_world_white_balance(img)
        print("DEBUG: Gray world white balance applied.")
        img_corrected = correct_lighting(img_wb)
        print("DEBUG: Lighting correction applied.")

        # --- Call Enhanced Placeholder AI modules ---
        # Note: These functions also have print statements for their simulation steps
        face_features = detect_face_features(img_corrected)
        tooth_analysis = segment_and_analyze_teeth(img_corrected)
        aesthetic_suggestion = aesthetic_shade_suggestion(face_features, tooth_analysis)
        print("DEBUG: Simulated AI modules executed.")
        # ------------------------------------

        height, width, _ = img_corrected.shape # Get width as well for sanity check
        print(f"DEBUG: Corrected image dimensions: Height={height}, Width={width}")

        # Ensure image has enough height for slicing
        # A common image height for a tooth could be hundreds of pixels.
        # If the image is tiny (e.g., less than 30 pixels high), slicing might result in empty arrays.
        min_height_for_slicing = 30 # Adjust based on expected smallest tooth image size
        if height < min_height_for_slicing:
            print(f"ERROR: Image height ({height} pixels) is too small for zonal slicing. Minimum required: {min_height_for_slicing} pixels. Cannot perform detailed shade detection.")
            return {
                "incisal": "Error", "middle": "Error", "cervical": "Error",
                "overall_ml_shade": "Error - Image Too Small",
                "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {}
            }

        incisal_zone = img_corrected[0:int(height*0.3), :, :]
        middle_zone = img_corrected[int(height*0.3):int(height*0.7), :, :]
        cervical_zone = img_corrected[int(height*0.7):height, :, :]
        print("DEBUG: Image zones sliced.")

        # Basic check to ensure zones are not empty before converting color space
        if incisal_zone.size == 0 or middle_zone.size == 0 or cervical_zone.size == 0:
            print(f"ERROR: One or more sliced image zones are empty. Check image dimensions and slicing logic. Incisal size: {incisal_zone.size}, Middle size: {middle_zone.size}, Cervical size: {cervical_zone.size}")
            return {
                "incisal": "Error", "middle": "Error", "cervical": "Error",
                "overall_ml_shade": "Error - Zone Empty",
                "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {}
            }

        incisal_lab = cv2.cvtColor(incisal_zone, cv2.COLOR_BGR2LAB)
        middle_lab = cv2.cvtColor(middle_zone, cv2.COLOR_BGR2LAB)
        cervical_lab = cv2.cvtColor(cervical_zone, cv2.COLOR_BGR2LAB)
        print("DEBUG: Zones converted to LAB color space (L range 0-255).")

        avg_incisal_l_255 = np.mean(incisal_lab[:,:,0])
        avg_middle_l_255 = np.mean(middle_lab[:,:,0])
        avg_cervical_l_255 = np.mean(cervical_lab[:,:,0])
        print(f"DEBUG: Average L values (0-255 scale): Incisal={avg_incisal_l_255:.2f}, Middle={avg_middle_l_255:.2f}, Cervical={avg_cervical_l_255:.2f}")

        # Normalize L values to 0-100 scale for ML prediction and rule-based mapping consistency
        # Assuming the training data's L values were in 0-100 scale.
        avg_incisal_l_100 = avg_incisal_l_255 / 2.55
        avg_middle_l_100 = avg_middle_l_255 / 2.55
        avg_cervical_l_100 = avg_cervical_l_255 / 2.55
        print(f"DEBUG: Average L values (0-100 scale): Incisal={avg_incisal_l_100:.2f}, Middle={avg_middle_l_100:.2f}, Cervical={avg_cervical_l_100:.2f}")


        if shade_classifier_model is not None:
            # Use 0-100 scaled L values for ML prediction
            features_for_ml_prediction = np.array([[avg_incisal_l_100, avg_middle_l_100, avg_cervical_l_100]])
            overall_ml_shade = shade_classifier_model.predict(features_for_ml_prediction)[0]
        else:
            overall_ml_shade = "Model Error"
            print("WARNING: AI model not loaded/trained. Cannot provide ML shade prediction.")

        detected_shades = {
            "incisal": map_l_to_shade_rule_based(avg_incisal_l_100), # Use 0-100 scaled for rule-based
            "middle": map_l_to_shade_rule_based(avg_middle_l_100),   # Use 0-100 scaled for rule-based
            "cervical": map_l_to_shade_rule_based(avg_cervical_l_100), # Use 0-100 scaled for rule-based
            "overall_ml_shade": overall_ml_shade,
            "face_features": face_features,
            "tooth_analysis": tooth_analysis,
            "aesthetic_suggestion": aesthetic_suggestion
        }

        print(f"DEBUG: Features for ML: {features_for_ml_prediction}")
        print(f"DEBUG: Predicted Overall Shade (ML): {overall_ml_shade}")
        print(f"DEBUG: Detected Shades Per Zone (Rule-based): {detected_shades}")
        return detected_shades

    except Exception as e:
        print(f"CRITICAL ERROR during shade detection: {e}") # Make this stand out
        traceback.print_exc() # This will print the full traceback to the console
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
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, txt=f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Detected Shades:", ln=True)
    pdf.set_font("Arial", size=12)
    if "overall_ml_shade" in shades and shades["overall_ml_shade"] != "N/A":
        pdf.cell(0, 7, txt=f"   - Overall AI Prediction : {shades['overall_ml_shade']}", ln=True)

    pdf.cell(0, 7, txt=f"   - Incisal Zone: {shades['incisal']}", ln=True)
    pdf.cell(0, 7, txt=f"   - Middle Zone: {shades['middle']}", ln=True)
    pdf.cell(0, 7, txt=f"   - Cervical Zone: {shades['cervical']}", ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=13)
    pdf.cell(0, 10, txt="Advanced AI Insights (Simulated):", ln=True)
    pdf.set_font("Arial", size=11)

    # Display Simulated Tooth Analysis
    tooth_analysis = shades.get("tooth_analysis", {})
    if tooth_analysis:
        pdf.cell(0, 7, txt="   -- Tooth Analysis --", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Overall Shade (Detailed): {tooth_analysis.get('simulated_overall_shade', 'N/A')}",
                 ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Condition: {tooth_analysis.get('tooth_condition', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Stain Presence: {tooth_analysis.get('stain_presence', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Decay Presence: {tooth_analysis.get('decay_presence', 'N/A')}", ln=True)
        # Ensure values exist before formatting
        l_val = tooth_analysis.get('overall_lab', {}).get('L', 'N/A')
        a_val = tooth_analysis.get('overall_lab', {}).get('a', 'N/A')
        b_val = tooth_analysis.get('overall_lab', {}).get('b', 'N/A')
        if all(isinstance(v, (int, float)) for v in [l_val, a_val, b_val]):
            pdf.cell(0, 7, txt=f"   - Simulated Overall LAB: L={l_val:.2f}, a={a_val:.2f}, b={b_val:.2f}", ln=True)
        else:
            pdf.cell(0, 7, txt=f"   - Simulated Overall LAB: L={l_val}, a={a_val}, b={b_val}", ln=True)
        # Removed the specific note as per user request
        # if tooth_analysis.get('notes'):
        #     pdf.multi_cell(0, 7, txt=f"   - Notes: {tooth_analysis.get('notes')}")

    pdf.ln(3)
    # Display Simulated Facial Features (including new skin/lip tone)
    face_features = shades.get("face_features", {})
    if face_features:
        pdf.cell(0, 7, txt="   -- Facial Aesthetics Analysis --", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Skin Tone: {face_features.get('skin_tone', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Lip Color: {face_features.get('lip_color', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Eye Contrast: {face_features.get('eye_contrast', 'N/A')}", ln=True)
        harmony_score = face_features.get('facial_harmony_score', 'N/A')
        if isinstance(harmony_score, (int, float)):
            pdf.cell(0, 7, txt=f"   - Simulated Facial Harmony Score: {harmony_score:.2f}", ln=True)
        else:
            pdf.cell(0, 7, txt=f"   - Simulated Facial Harmony Score: {harmony_score}", ln=True)
        # Removed the specific note as per user request
        # if face_features.get('notes'):
        #     pdf.multi_cell(0, 7, txt=f"   - Notes: {face_features.get('notes')}")

    pdf.ln(3)
    # Display Simulated Aesthetic Suggestion
    aesthetic_suggestion = shades.get("aesthetic_suggestion", {})
    if aesthetic_suggestion:
        pdf.cell(0, 7, txt="   -- Aesthetic Shade Suggestion --", ln=True)
        pdf.cell(0, 7, txt=f"   - Suggested Shade: {aesthetic_suggestion.get('suggested_aesthetic_shade', 'N/A')}",
                 ln=True)
        pdf.cell(0, 7, txt=f"   - Confidence: {aesthetic_suggestion.get('aesthetic_confidence', 'N/A')}", ln=True)
        pdf.multi_cell(0, 7, txt=f"   - Notes: {aesthetic_suggestion.get('recommendation_notes', 'N/A')}")

    pdf.ln(10)

    try:
        if os.path.exists(image_path):
            pdf.cell(0, 10, txt="Uploaded Image:", ln=True)
            if pdf.get_y() > 200:
                pdf.add_page()
            img_cv = cv2.imread(image_path)
            if img_cv is not None:
                h_img, w_img, _ = img_cv.shape
                max_w_pdf = 180
                w_pdf = min(w_img, max_w_pdf)
                h_pdf = h_img * (w_pdf / w_img)

                if pdf.get_y() + h_pdf + 10 > pdf.h - pdf.b_margin:
                    pdf.add_page()

                pdf.image(image_path, x=pdf.get_x(), y=pdf.get_y(), w=w_pdf, h=h_pdf)
                pdf.ln(h_pdf + 10)
            else:
                pdf.cell(0, 10, txt="Note: Image could not be loaded for embedding.", ln=True)

        else:
            pdf.cell(0, 10, txt="Note: Uploaded image file not found for embedding.", ln=True)
    except Exception as e:
        print(f"Error adding image to PDF: {e}")
        pdf.cell(0, 10, txt="Note: An error occurred while embedding the image in the report.", ln=True)

    pdf.set_font("Arial", 'I', size=9)  # Disclaimer in italics
    pdf.multi_cell(0, 6,
                   txt="DISCLAIMER: This report is based on simulated AI analysis for demonstration purposes only. It is not intended for clinical diagnosis, medical advice, or professional cosmetic planning. Always consult with a qualified dental or medical professional for definitive assessment, diagnosis, and treatment.",
                   align='C')
    pdf.output(filepath)


# ===============================================
# 5. ROUTES (Adapted for Firestore)
# ===============================================
@app.route('/')
def home():
    """Renders the home/landing page."""
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login (Simulated for Canvas)."""
    # If user is already "logged in" (i.e., not anonymous), redirect
    if g.user and 'anonymous' not in g.user['id']:
        flash(f"You are already logged in as {g.user['username']}.", 'info')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']  # In a real app, hash and check this
        error = None

        if not username or not password:
            error = 'Username and password are required.'

        if error is None:
            # Simulate user lookup/creation and set session
            simulated_user_id = 'user_' + username.lower().replace(' ', '_')
            session['user_id'] = simulated_user_id
            session['user'] = {'id': simulated_user_id, 'username': username}
            flash(f'Simulated login successful for {username}!', 'success')
            print(f"DEBUG: Simulated login for user: {username} (ID: {session['user_id']})")
            return redirect(url_for('dashboard'))
        flash(error, 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles user registration (Simulated for Canvas)."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            # Simulate user creation - no actual user database interaction here
            # In a real app, you'd add user to Firebase Auth/Firestore
            flash(f"Simulated registration successful for {username}. You can now log in!", 'success')
            return redirect(url_for('login'))
        flash(error, 'danger')

    return render_template('register.html')


@app.route('/logout')
def logout(): # login_required decorator removed as we are directly modifying session here
    """Handles user logout."""
    # Clear session and reset to an anonymous user
    session.clear() # Clears all session data
    # The next before_request will re-initialize g.user_id and g.user to an anonymous one.
    flash('You have been logged out.', 'info')
    print(f"DEBUG: User logged out. Session cleared.")
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the user dashboard, displaying past reports."""
    # Fetch reports specific to the current user (simulated)
    # Path: artifacts/{app_id}/users/{user_id}/reports
    reports_path = ['artifacts', app_id, 'users', g.firestore_user_id, 'reports']
    user_reports = get_firestore_documents_in_collection(reports_path)
    # Sort by date in descending order, if available
    user_reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    # Pass the current date formatted for the HTML date input
    current_date_formatted = datetime.now().strftime('%Y-%m-%d')

    return render_template('dashboard.html',
                           reports=user_reports,
                           user=g.user,
                           current_date=current_date_formatted) # Pass current_date here


@app.route('/save_patient_data', methods=['POST'])
@login_required
def save_patient_data():
    """Handles saving new patient records to Firestore and redirects to image upload page."""
    op_number = request.form['op_number']
    patient_name = request.form['patient_name']
    age = request.form['age']
    sex = request.form['sex']
    record_date = request.form['date']
    user_id = g.user['id'] # The current Flask session user ID

    # Firestore path for patient data: /artifacts/{appId}/users/{userId}/patients
    patients_collection_path = ['artifacts', app_id, 'users', user_id, 'patients']

    # Simulate checking if OP Number exists for this user in Firestore
    existing_patients = get_firestore_documents_in_collection(patients_collection_path, query_filters={'op_number': op_number, 'user_id': user_id})
    if existing_patients:
        flash('OP Number already exists for another patient under your account. Please use a unique OP Number or select from recent entries.', 'error')
        return redirect(url_for('dashboard'))

    try:
        patient_data = {
            'user_id': user_id,
            'op_number': op_number,
            'patient_name': patient_name,
            'age': int(age),
            'sex': sex,
            'record_date': record_date,
            'created_at': datetime.now().isoformat() # ISO format for easy sorting
        }

        # Add patient data to Firestore
        add_firestore_document(patients_collection_path, patient_data)

        flash('Patient record saved successfully (to Firestore)! Now upload an image.', 'success')
        return redirect(url_for('upload_page', op_number=op_number))
    except Exception as e:
        flash(f'Error saving patient record to Firestore: {e}', 'error')
        return redirect(url_for('dashboard'))


@app.route('/upload_page/<op_number>')
@login_required
def upload_page(op_number):
    """Renders the dedicated image upload page for a specific patient."""
    user_id = g.user['id']

    # Fetch patient from Firestore
    patients_collection_path = ['artifacts', app_id, 'users', user_id, 'patients']
    patient = None
    all_patients = get_firestore_documents_in_collection(patients_collection_path, query_filters={'op_number': op_number, 'user_id': user_id})
    if all_patients:
        patient = all_patients[0] # Assuming op_number is unique per user

    if patient is None:
        flash('Patient not found or unauthorized access.', 'error')
        return redirect(url_for('dashboard'))

    return render_template('upload_page.html', op_number=op_number, patient_name=patient['patient_name'])


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file(): # Renamed from 'upload' to 'upload_file' for clarity and consistency
    """Handles image upload, shade detection, and PDF report generation."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        patient_name = request.form.get('patient_name', 'Unnamed Patient') # Get patient_name from form

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            # Generate a unique filename using UUID and original extension
            file_ext = os.path.splitext(filename)[1] # Get file extension (e.g., .jpg)
            unique_filename = str(uuid.uuid4()) + file_ext # Combine UUID with extension
            
            original_image_path = os.path.join(UPLOAD_FOLDER, unique_filename) # Use unique_filename here
            file.save(original_image_path)
            flash('Image uploaded successfully!', 'success')

            # Perform shade detection
            detected_shades = detect_shades_from_image(original_image_path)

            if detected_shades["overall_ml_shade"] == "Error":
                flash("Error processing image for shade detection.", 'danger')
                return redirect(url_for('upload_page', op_number=request.form.get('op_number'))) # Redirect back to upload page

            # Generate PDF report
            report_filename = f"report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_filepath = os.path.join(REPORT_FOLDER, report_filename)
            generate_pdf_report(patient_name, detected_shades, original_image_path, report_filepath)
            flash('PDF report generated!', 'success')

            # Prepare analysis date for the HTML template
            formatted_analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Save report metadata to simulated Firestore
            report_data = {
                'patient_name': patient_name,
                'op_number': request.form.get('op_number'), # Ensure OP number is saved with report
                'original_image': unique_filename, # Store the unique filename in DB
                'report_filename': report_filename,
                'detected_shades': detected_shades, # Store all detected shades
                'timestamp': datetime.now().isoformat(),
                'user_id': g.firestore_user_id # Link to the current user
            }
            reports_collection_path = ['artifacts', app_id, 'users', g.firestore_user_id, 'reports']
            add_firestore_document(reports_collection_path, report_data)

            # Pass data to the report display template
            return render_template('report.html',
                                   patient_name=patient_name,
                                   shades=detected_shades,
                                   image_filename=unique_filename, # Pass unique filename to template
                                   report_filename=report_filename,
                                   analysis_date=formatted_analysis_date) # Pass formatted date

    # Handle GET request for upload_file route (if a user directly navigates to /upload)
    # This route is usually accessed via /upload_page/<op_number> which has proper context.
    # We can default to redirecting to dashboard or providing a basic form if accessed directly.
    flash("Please select a patient from the dashboard to upload an image.", 'info')
    return redirect(url_for('dashboard')) # Redirect if accessed directly without patient context


@app.route('/download_report/<filename>')
@login_required
def download_report(filename):
    """Allows downloading of generated PDF reports."""
    # In a real app, you'd ensure this report belongs to the current user's accessible reports.
    # For this simulation, we'll allow download if the file exists.
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)

# Main entry point
if __name__ == '__main__':
    # Ensure model is loaded/trained on startup
    if shade_classifier_model is None:
        print("CRITICAL: Machine Learning model could not be loaded or trained. Shade prediction will not work.")
    app.run(debug=True)
