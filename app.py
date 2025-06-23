import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash, g
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF
from datetime import datetime
import cv2 # For image processing
import numpy as np # For numerical operations with images

# --- NEW IMPORTS FOR AI (Machine Learning) INTEGRATION ---
from sklearn.neighbors import KNeighborsClassifier
import joblib # For saving/loading machine learning models
import pandas as pd # ADDED THIS IMPORT FOR CSV HANDLING
# ---------------------------------------------------------

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
# 4. CORE HELPER FUNCTIONS (Image Correction, Shade Detection, PDF Generation)
# ===============================================

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
    A.pplies CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting correction.
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

# --- AI Model Setup (Loading Data from CSV & Training/Loading) ---
model_filename = 'shade_classifier_model.pkl'
shade_classifier_model = None
dataset_filename = 'tooth_shades_simulated.csv' # Define your dataset file

# Attempt to load the model. If not found, train a new one using the CSV data.
try:
    shade_classifier_model = joblib.load(model_filename)
    print(f"DEBUG: Loaded pre-trained shade model from {model_filename}")
except FileNotFoundError:
    print(f"DEBUG: Training new simple shade model (from {dataset_filename}) as no existing model found.")
    
    # Load data from CSV using pandas
    try:
        df = pd.read_csv(dataset_filename)
        # Features (X) are the lightness values (incisal_l, middle_l, cervical_l)
        X_train = df[['incisal_l', 'middle_l', 'cervical_l']].values
        # Labels (y) are the overall shades
        y_train = df['overall_shade'].values
        
        # Ensure we have data to train
        if X_train.shape[0] == 0:
            raise ValueError(f"No data found in {dataset_filename} for training.")

        # Using KNeighborsClassifier as a simple example.
        # In a real scenario, you might use a more complex model (e.g., RandomForest, SVM)
        # or a deep learning model (CNN) if you have enough data.
        shade_classifier_model = KNeighborsClassifier(n_neighbors=3)
        shade_classifier_model.fit(X_train, y_train)
        joblib.dump(shade_classifier_model, model_filename) # Save the trained model
        print(f"DEBUG: Model trained and saved to {model_filename}")

    except FileNotFoundError:
        print(f"ERROR: Dataset file '{dataset_filename}' not found. Cannot train model. Please create it.")
        # Fallback if dataset is missing, or raise an error to stop execution
        shade_classifier_model = None # Ensure model is None if data missing
    except Exception as e:
        print(f"ERROR: Failed to load/process dataset or train model: {e}")
        shade_classifier_model = None # Ensure model is None if training fails

# --- End AI Model Setup ---


def detect_shades_from_image(image_path):
    """
    Performs lighting correction, white balance, extracts features,
    and then uses the pre-trained ML model for overall tooth shade detection.
    Also, provides rule-based shades for individual zones for UI consistency.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return {"incisal": "N/A", "middle": "N/A", "cervical": "N/A", "overall_ml_shade": "N/A"}

        # --- Apply Image Pre-processing ---
        # 1. Apply Gray World White Balance to correct color casts
        img_wb = gray_world_white_balance(img)
        
        # 2. Apply CLAHE for contrast and brightness normalization
        img_corrected = correct_lighting(img_wb)

        height, _, _ = img_corrected.shape # Use the corrected image for feature extraction
        
        # Define approximate regions for zones (top 30%, middle 40%, bottom 30%)
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

        # --- AI-BASED OVERALL SHADE PREDICTION ---
        # Prepare the features (average L values of zones) for the ML model
        # The model expects a 2D array, even for a single prediction
        features_for_ml_prediction = np.array([[avg_incisal_l, avg_middle_l, avg_cervical_l]])
        
        # Use the pre-trained ML model to predict the overall shade
        # This will be the AI's "diagnosis"
        # Ensure the model exists before predicting
        if shade_classifier_model is not None:
            overall_ml_shade = shade_classifier_model.predict(features_for_ml_prediction)[0]
        else:
            overall_ml_shade = "Model Error"
            print("WARNING: AI model not loaded/trained. Cannot provide ML shade prediction.")
        # ------------------------------------------

        # --- RULE-BASED PER-ZONE SHADE MAPPING (for detailed report display) ---
        # This function is kept to provide per-zone shade values as expected by report.html
        # In a more advanced AI system, the ML model itself might output per-zone predictions.
        def map_l_to_shade_rule_based(l_value):
            if l_value > 75: return "B1" # Very light
            elif l_value > 70: return "A1"
            elif l_value > 65: return "A2"
            elif l_value > 60: return "B2"
            elif l_value > 55: return "C2"
            elif l_value > 50: return "A3"
            else: return "C3" # Darker

        detected_shades = {
            "incisal": map_l_to_shade_rule_based(avg_incisal_l),
            "middle": map_l_to_shade_rule_based(avg_middle_l),
            "cervical": map_l_to_shade_rule_based(avg_cervical_l),
            "overall_ml_shade": overall_ml_shade # Include the AI's overall prediction
        }
        
        print(f"DEBUG: Features for ML: {features_for_ml_prediction}")
        print(f"DEBUG: Predicted Overall Shade (ML): {overall_ml_shade}")
        print(f"DEBUG: Detected Shades Per Zone (Rule-based): {detected_shades}")
        return detected_shades

    except Exception as e:
        print(f"Error during shade detection: {e}")
        return {"incisal": "Error", "middle": "Error", "cervical": "Error", "overall_ml_shade": "Error"}


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
        pdf.cell(0, 7, txt=f"  - Overall AI Prediction: {shades['overall_ml_shade']}", ln=True)
    
    # Display per-zone shades
    pdf.cell(0, 7, txt=f"  - Incisal Zone: {shades['incisal']}", ln=True)
    pdf.cell(0, 7, txt=f"  - Middle Zone: {shades['middle']}", ln=True)
    pdf.cell(0, 7, txt=f"  - Cervical Zone: {shades['cervical']}", ln=True)
    
    pdf.ln(10)

    try:
        if os.path.exists(image_path):
            pdf.cell(0, 10, txt="Uploaded Image:", ln=True)
            if pdf.get_y() > 200:
                pdf.add_page()
            pdf.image(image_path, x=10, y=pdf.get_y(), w=100)
            pdf.ln(100)
    except Exception as e:
        print(f"Error adding image to PDF: {e}")
        pdf.cell(0, 10, txt="Note: Image could not be embedded in the report.", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt="This report provides an estimated tooth shade.", ln=True)
    pdf.cell(0, 10, txt="Please consult with a dental professional for definitive assessment.", ln=True)

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
        detected_shades = detect_shades_from_image(filepath)

        # Generate PDF Report
        report_filename = f"{op_number}_report_{os.path.splitext(unique_filename)[0]}.pdf"
        report_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)
        
        generate_pdf_report(patient['patient_name'], detected_shades, filepath, report_path)

        flash(f'Image uploaded and shades detected for {patient["patient_name"]}. Report generated!', 'success')
        # Pass the 'overall_ml_shade' to the template if you want to display it
        return render_template(
            'report.html',
            user_name=patient['patient_name'],
            shades=detected_shades,
            report_filename=report_filename,
            overall_ml_shade=detected_shades.get('overall_ml_shade', 'N/A') # Pass overall ML shade
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
    