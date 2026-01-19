from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================
# Flask App Setup
# ============================================

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here'

# Folder for uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ============================================
# ROUTES
# ============================================

@app.route('/')
@app.route('/home')
def home():
    """Render Home Page"""
    return render_template('home.html')


@app.route('/login')
def login():
    """Render Login Page"""
    return render_template('login.html')


@app.route('/login-submit', methods=['POST'])
def login_submit():
    """Handle Login"""
    username = request.form.get('username')
    password = request.form.get('password')

    VALID_USERNAME = 'name@example.com'
    VALID_PASSWORD = 'password123'

    if username == VALID_USERNAME and password == VALID_PASSWORD:
        session['username'] = username
        flash('Login Successful! Welcome to the Fraud Detection Dashboard.', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid Username or Password.', 'error')
        return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    """Upload Page"""
    if 'username' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    return render_template('upload.html')
@app.route('/upload', methods=['GET'])
def upload_page():
    if 'username' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV Upload and Redirect to Preview"""
    if 'username' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))

    file = request.files.get('file')

    if not file or file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('upload_page'))

    if not file.filename.endswith('.csv'):
        flash('Only CSV files are allowed.', 'error')
        return redirect(url_for('upload_page'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        session['uploaded_file'] = filepath
        flash('File uploaded successfully! Previewing sample data...', 'success')
        return redirect(url_for('preview_data'))
    except Exception as e:
        flash(f'Error reading CSV file: {e}', 'error')
        return redirect(url_for('upload_page'))


@app.route('/preview')
def preview_data():
    """Show 5000 sample rows from uploaded CSV"""
    if 'username' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))

    filepath = session.get('uploaded_file')
    if not filepath or not os.path.exists(filepath):
        flash('No uploaded file found.', 'error')
        return redirect(url_for('upload_page'))

    try:
        df = pd.read_csv(filepath)
        sample_df = df.head(5000)
        table_html = sample_df.to_html(classes='table table-striped table-bordered', index=False)
        return render_template('preview.html', table=table_html, filename=os.path.basename(filepath))
    except Exception as e:
        flash(f'Error displaying preview: {e}', 'error')
        return redirect(url_for('upload_page'))


@app.route('/fraudform')
def fraudform():
    """Render Model Prediction Form"""
    if 'username' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    return render_template('fraudform.html')


# ============================================
# Load Pre-Trained Models
# ============================================

try:
    xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
    stack_model = pickle.load(open('stacking_model.pkl', 'rb'))
except Exception as e:
    xgb_model = None
    stack_model = None
    print(f"⚠️ Warning: Model files not found or failed to load: {e}")


# ============================================
# Fraud Prediction Route
# ============================================
@app.route('/predict', methods=['POST'])
def predict_fraud():
    """Predict fraud by checking CSV match + ML model (XGB/Stack) result."""
    try:
        print("\n🚀 ENTERED /PREDICT FUNCTION 🚀")

        # ✅ Get form data
        transaction_amount = float(request.form['transaction_amount'])
        payment_method = request.form['payment_method']
        product_category = request.form['product_category']
        device_used = request.form['device_used']
        quantity = int(request.form['quantity'])
        account_age = int(request.form['account_age'])
        transaction_hour = int(request.form['transaction_hour'])
        model_type = request.form['model_type'].lower()

        # ✅ Create DataFrame for input
        input_data = pd.DataFrame([{
            'Transaction Amount': transaction_amount,
            'Payment Method': payment_method,
            'Product Category': product_category,
            'Device Used': device_used,
            'Quantity': quantity,
            'Account Age (Days)': account_age,
            'Transaction Hour': transaction_hour
        }])

        # ✅ Check if uploaded CSV is available
        filepath = session.get('uploaded_file')
        if not filepath or not os.path.exists(filepath):
            return render_template('result.html',
                                   result="🚨 Fraud Detected!",
                                   model_used=model_type.upper(),
                                   message="⚠️ No uploaded CSV found for comparison.")

        # ✅ Read CSV
        df = pd.read_csv(filepath)
        print("Uploaded CSV columns:", df.columns.tolist())

        # ✅ Rename columns if needed
        df = df.rename(columns={
            'TransactionAmount': 'Transaction Amount',
            'PaymentType': 'Payment Method',
            'ProductCategory': 'Product Category',
            'DeviceType': 'Device Used',
            'AccountAge': 'Account Age (Days)',
            'TransactionHour': 'Transaction Hour',
            'IsFraudulent': 'IsFraudulent'
        })

        # ✅ Clean and normalize
        for col in ['Payment Method', 'Product Category', 'Device Used']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

        payment_method = payment_method.strip().lower()
        product_category = product_category.strip().lower()
        device_used = device_used.strip().lower()

        # ✅ Numeric conversions
        for col in ['Transaction Amount', 'Quantity', 'Account Age (Days)', 'Transaction Hour']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # ✅ Exact match condition
        match = (
            (np.isclose(df['Transaction Amount'], transaction_amount, atol=0.01)) &
            (df['Payment Method'] == payment_method) &
            (df['Product Category'] == product_category) &
            (df['Device Used'] == device_used) &
            (df['Quantity'] == quantity) &
            (df['Account Age (Days)'] == account_age) &
            (df['Transaction Hour'] == transaction_hour)
        )

        # ✅ Prepare input for model
        model_input = input_data.copy()
        categorical_cols = ['Payment Method', 'Product Category', 'Device Used']
        model_input = pd.get_dummies(model_input, columns=categorical_cols)

        model = xgb_model if model_type == "xgb" else stack_model
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            model_features = model_input.columns

        for col in model_features:
            if col not in model_input.columns:
                model_input[col] = 0

        model_input = model_input[model_features]

        # ✅ Model prediction
        prediction = int(model.predict(model_input)[0])
        model_result = "🚨 Fraud Detected!" if prediction == 1 else "✅ Legitimate Transaction"

        # ✅ CASE 1 — Exact Match Found
        if not df[match].empty:
            matched_row = df[match].iloc[0]
            print("\n✅ Exact Match Found in CSV:", matched_row.to_dict())

            csv_label = int(matched_row.get('IsFraudulent', 0))

            # ✅ Final logic: if either model or CSV says fraud → fraud
            if csv_label == 1 or prediction == 1:
                result = "🚨 Fraud Detected!"
                message = (f"⚠️ Fraud found! CSV label: {csv_label}, "
                           f"{model_type.upper()} model prediction: {prediction}")
            else:
                result = "✅ Legitimate Transaction"
                message = f"✔️ Both CSV and model confirm it's legitimate (CSV={csv_label}, Model={prediction})."

        # ✅ CASE 2 — No Exact Match Found
        else:
            result = "🚨 Fraud Detected!"
            message = "⚠️ No matching record in CSV. Marked as Fraud by system safeguard."

        print("=== DEBUG INFO END ===\n")
        return render_template('result.html',
                               result=result,
                               model_used=model_type.upper(),
                               message=message)

    except Exception as e:
        print("❌ Error in /predict:", str(e))
        return render_template('result.html',
                               result="🚨 Fraud Detected!",
                               model_used="N/A",
                               message=f"❌ Error: {str(e)}")

@app.route('/performance')
def performance():

    # Load dataset again (required for test split)
    df = pd.read_csv("ecommerce_fraud_data.csv")
    target_col = "IsFraudulent"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Load Label Encoders
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    for col in label_encoders:
        X[col] = label_encoders[col].transform(X[col].astype(str))

    # Train/Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------------
    # 1️⃣ XGBOOST METRICS
    # ----------------------------
    xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
    y_pred_xgb = xgb_model.predict(X_test)

    xgb_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred_xgb) * 100, 2),
        "precision": round(precision_score(y_test, y_pred_xgb, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred_xgb, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred_xgb, zero_division=0), 4),
        "cm": confusion_matrix(y_test, y_pred_xgb).tolist()
    }

    # ----------------------------
    # 2️⃣ STACKING MODEL METRICS
    # ----------------------------
    stacking_model = pickle.load(open("stacking_model.pkl", "rb"))
    y_pred_stack = stacking_model.predict(X_test)

    stack_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred_stack) * 100, 2),
        "precision": round(precision_score(y_test, y_pred_stack, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred_stack, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred_stack, zero_division=0), 4),
        "cm": confusion_matrix(y_test, y_pred_stack).tolist()
    }

    return render_template(
        "performance.html",
        xgb=xgb_metrics,
        stack=stack_metrics
    )
@app.route('/performance/charts')
def performance_charts():

    # REUSE SAME DATA & MODELS
    df = pd.read_csv("ecommerce_fraud_data.csv")
    target_col = "IsFraudulent"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    for col in label_encoders:
        X[col] = label_encoders[col].transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Load models
    xgb = pickle.load(open("xgb_model.pkl", "rb"))
    st = pickle.load(open("stacking_model.pkl", "rb"))

    # Predictions
    yx = xgb.predict(X_test)
    ys = st.predict(X_test)

    # Metrics
    import os
    chart_dir = "static/charts"
    os.makedirs(chart_dir, exist_ok=True)

    # Accuracy Chart
    plt.figure(figsize=(5,4))
    plt.bar(["XGBoost","Stacking"], 
            [accuracy_score(y_test, yx), accuracy_score(y_test, ys)],
            color=["blue","green"])
    plt.title("Accuracy Comparison")
    acc_path = os.path.join(chart_dir, "acc_chart.png")
    plt.savefig(acc_path); plt.close()

    # Metric Chart (Precision, Recall, F1)
    plt.figure(figsize=(6,4))
    labels = ["Precision","Recall","F1 Score"]
    xgb_vals = [
        precision_score(y_test, yx), 
        recall_score(y_test, yx), 
        f1_score(y_test, yx)
    ]
    st_vals = [
        precision_score(y_test, ys), 
        recall_score(y_test, ys), 
        f1_score(y_test, ys)
    ]
    x = range(3)
    plt.bar(x, xgb_vals, width=0.4, label="XGBoost")
    plt.bar([i+0.4 for i in x], st_vals, width=0.4, label="Stacking")
    plt.xticks([i+0.2 for i in x], labels)
    plt.title("Metric Comparison")
    plt.legend()
    metric_path = os.path.join(chart_dir, "metric_chart.png")
    plt.savefig(metric_path); plt.close()

    return render_template(
            "charts.html",
            acc_chart="charts/acc_chart.png",
            metric_chart="charts/metric_chart.png"
    )
@app.route('/performance/confusion')
def performance_confusion():

    df = pd.read_csv("ecommerce_fraud_data.csv")
    target_col = "IsFraudulent"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    for col in label_encoders:
        X[col] = label_encoders[col].transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    xgb = pickle.load(open("xgb_model.pkl", "rb"))
    st = pickle.load(open("stacking_model.pkl", "rb"))

    yx = xgb.predict(X_test)
    ys = st.predict(X_test)

    # Confusion matrices
    import os
    chart_dir = "static/charts"
    os.makedirs(chart_dir, exist_ok=True)

    # XGB
    plt.figure(figsize=(4,3))
    sns.heatmap(confusion_matrix(y_test, yx), annot=True, cmap="Blues", fmt="d")
    xgb_cm = os.path.join(chart_dir, "xgb_cm.png")
    plt.savefig(xgb_cm); plt.close()

    # STACKING
    plt.figure(figsize=(4,3))
    sns.heatmap(confusion_matrix(y_test, ys), annot=True, cmap="Greens", fmt="d")
    st_cm = os.path.join(chart_dir, "st_cm.png")
    plt.savefig(st_cm); plt.close()

    return render_template(
        "confusion.html",
        xgb_cm="charts/xgb_cm.png",
        st_cm="charts/st_cm.png"
    )

# ============================================
# Logout
# ============================================

@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('username', None)
    session.pop('uploaded_file', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ============================================
# Run App
# ============================================

if __name__ == '__main__':
    app.run(debug=True)
