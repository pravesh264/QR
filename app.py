import os
import cv2
import numpy as np
from flask import Flask, render_template, request
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Paths to dataset and pre-generated files
FIRST_PRINTS_DIR = "D:/Projects/QRGuardian/dataset/first_prints"
SECOND_PRINTS_DIR = "D:/Projects/QRGuardian/dataset/second_prints"
STATIC_DIR = "D:/Projects/QRGuardian/static/"
ROOT_DIR = "D:/Projects/QRGuardian/"  # Where combined_training.py saves plots by default

# Load pre-trained models
svm_model = joblib.load('D:/Projects/QRGuardian/svm_model.pkl')
scaler = joblib.load('D:/Projects/QRGuardian/scaler.pkl')
cnn_model = tf.keras.models.load_model('D:/Projects/QRGuardian/cnn_model.keras')

# Feature extraction for inference
def extract_features(img_path, for_svm=True):
    """Extract features for SVM or CNN inference."""
    if for_svm:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None, None
        img = cv2.resize(img, (224, 224))
        edges = cv2.Canny(img, 100, 200)
        edge_intensity = np.mean(edges)
        texture_variance = np.std(img)
        return edge_intensity, texture_variance, edges
    else:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

# Dataset stats
stats = {
    'first_prints': len(os.listdir(FIRST_PRINTS_DIR)),
    'second_prints': len(os.listdir(SECOND_PRINTS_DIR)),
    'total': len(os.listdir(FIRST_PRINTS_DIR)) + len(os.listdir(SECOND_PRINTS_DIR))
}

# Model performance metrics from latest training run
performance_metrics = {
    'svm': {'accuracy': 0.79, 'precision': 0.82, 'recall': 0.73, 'f1_score': 0.77},
    'cnn': {'accuracy': 0.80, 'precision': 0.71, 'recall': 1.00, 'f1_score': 0.83}
}

# Plotting helper
def plot_to_base64(fig):
    """Convert a Matplotlib figure to base64 for HTML rendering."""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Generate or load performance plots with fallback
def generate_performance_plots():
    """Load static plots from combined_training.py output with fallback."""
    def load_plot(filename, fallback_text="Plot not available"):
        # Try static/ first, then root directory
        filepath = os.path.join(STATIC_DIR, filename)
        if not os.path.exists(filepath):
            filepath = os.path.join(ROOT_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        else:
            # Fallback placeholder
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(0.5, 0.5, fallback_text, ha='center', va='center')
            ax.axis('off')
            plot_data = plot_to_base64(fig)
            plt.close(fig)
            return plot_data

    dataset_plot = load_plot('dataset_comparison.png')
    feature_plot = load_plot('feature_scatter.png')
    eval_plot = load_plot('confusion_matrices.png')
    return dataset_plot, feature_plot, eval_plot

@app.route('/', methods=['GET', 'POST'])
def upload():
    """Handle QR code upload and display initial prediction."""
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        edge_intensity, texture_variance, edges = extract_features(filepath, for_svm=True)
        if edge_intensity is None:
            return "Error: Could not process image", 400
        
        cnn_img = extract_features(filepath, for_svm=False)
        if cnn_img is None:
            return "Error: Could not process image", 400
        cnn_input = np.expand_dims(cnn_img / 255.0, axis=0)
        
        svm_features_scaled = scaler.transform([[edge_intensity, texture_variance]])
        svm_prob = svm_model.predict_proba(svm_features_scaled)[0]
        cnn_prob = cnn_model.predict(cnn_input)[0]
        svm_result = 'First Print' if svm_prob[0] > svm_prob[1] else 'Second Print'
        cnn_result = 'First Print' if cnn_prob < 0.5 else 'Second Print'
        svm_confidence = max(svm_prob)
        cnn_confidence = 1 - cnn_prob if cnn_prob < 0.5 else cnn_prob
        
        feature_path = os.path.join(app.config['UPLOAD_FOLDER'], 'edges_' + file.filename)
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap_' + file.filename)
        cv2.imwrite(feature_path, edges)
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, heatmap)
        
        security_note = "Authentic" if svm_result == 'First Print' and cnn_result == 'First Print' else "Possible Counterfeit"
        return render_template('result.html', svm_result=svm_result, svm_confidence=svm_confidence,
                             cnn_result=cnn_result, cnn_confidence=cnn_confidence, security_note=security_note,
                             img_path=file.filename, feature_path='edges_' + file.filename, heatmap_path='heatmap_' + file.filename)
    return render_template('upload.html')

@app.route('/performance')
def performance():
    """Display dataset stats, feature analysis, and model evaluation."""
    dataset_plot, feature_plot, eval_plot = generate_performance_plots()
    return render_template('performance.html', stats=stats, dataset_plot=dataset_plot, 
                          feature_plot=feature_plot, eval_plot=eval_plot, metrics=performance_metrics)

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    """Handle verification with a second QR code."""
    if request.method == 'POST':
        initial_result = request.form['initial_result']
        initial_confidence = float(request.form['initial_confidence'])
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        edge_intensity, texture_variance, edges = extract_features(filepath, for_svm=True)
        if edge_intensity is None:
            return "Error: Could not process image", 400
        
        cnn_img = extract_features(filepath, for_svm=False)
        if cnn_img is None:
            return "Error: Could not process image", 400
        cnn_input = np.expand_dims(cnn_img / 255.0, axis=0)
        
        svm_features_scaled = scaler.transform([[edge_intensity, texture_variance]])
        svm_prob = svm_model.predict_proba(svm_features_scaled)[0]
        cnn_prob = cnn_model.predict(cnn_input)[0]
        svm_result = 'First Print' if svm_prob[0] > svm_prob[1] else 'Second Print'
        cnn_result = 'First Print' if cnn_prob < 0.5 else 'Second Print'
        svm_confidence = max(svm_prob)
        cnn_confidence = 1 - cnn_prob if cnn_prob < 0.5 else cnn_prob
        
        final_result = 'First Print' if svm_result == initial_result and cnn_result == initial_result else 'Inconclusive'
        final_confidence = (initial_confidence + svm_confidence + cnn_confidence) / 3
        
        feature_path = os.path.join(app.config['UPLOAD_FOLDER'], 'edges_' + file.filename)
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap_' + file.filename)
        cv2.imwrite(feature_path, edges)
        cv2.imwrite(heatmap_path, cv2.applyColorMap(edges, cv2.COLORMAP_JET))
        
        security_note = "Consistent" if final_result == 'First Print' else "Inconsistent"
        return render_template('verify_result.html', initial_result=initial_result, initial_confidence=initial_confidence,
                             svm_result=svm_result, svm_confidence=svm_confidence, cnn_result=cnn_result,
                             cnn_confidence=cnn_confidence, final_result=final_result, final_confidence=final_confidence,
                             security_note=security_note, img_path=file.filename, feature_path='edges_' + file.filename,
                             heatmap_path='heatmap_' + file.filename)
    return render_template('verify.html')

if __name__ == '__main__':
    app.run(debug=True)