import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import torch
from models.model_architecture import StackedLSTM_MultiPooling
from utils.data_processing import preprocess_ecg_data
from utils.visualization import generate_ecg_interactive, generate_statistics_plotly

app = Flask(__name__)

# Load the trained model
model = StackedLSTM_MultiPooling(
    input_size=12,  # Num channels
    hidden_size=64,
    num_layers=2,
    num_classes=2
)
model.load_state_dict(
    torch.load('models/ecg_model.pth', map_location=torch.device('cpu'))
)
model.eval()

# Upload folder config
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Reference stats per-channel for normal and AF (replace placeholder values with real reference data)
# Typical 12-lead column order: ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
reference_stats = {
    'normal': {
        'I':   {'mean': -49.30, 'median': -25.00, 'std': 110.41},
        'II':  {'mean': -70.60, 'median': -41.00, 'std': 142.33},
        'III': {'mean': -21.29, 'median': -14.00, 'std': 97.13},
        'aVR': {'mean': 59.95,  'median': 33.00,  'std': 117.76},
        'aVL': {'mean': -14.01, 'median': -5.00,  'std': 75.82},
        'aVF': {'mean': -45.94, 'median': -28.00, 'std': 108.62},
        'V1':  {'mean': 7.75,   'median': -9.00,  'std': 195.39},
        'V2':  {'mean': -73.32, 'median': -43.00, 'std': 247.94},
        'V3':  {'mean': -70.20, 'median': -45.00, 'std': 203.57},
        'V4':  {'mean': -78.47, 'median': -42.00, 'std': 213.57},
        'V5':  {'mean': -77.61, 'median': -37.00, 'std': 218.63},
        'V6':  {'mean': -71.34, 'median': -32.00, 'std': 225.84},
    },
    'af': {
        'I':   {'mean': -32.37, 'median': -18.00, 'std': 111.76},
        'II':  {'mean': -59.42, 'median': -39.00, 'std': 140.16},
        'III': {'mean': -27.05, 'median': -21.00, 'std': 120.40},
        'aVR': {'mean': 45.90,  'median': 28.00,  'std': 111.55},
        'aVL': {'mean': -2.66,  'median': 2.00,   'std': 92.64},
        'aVF': {'mean': -43.24, 'median': -31.00, 'std': 118.10},
        'V1':  {'mean': 2.67,   'median': -8.00,  'std': 159.12},
        'V2':  {'mean': -52.54, 'median': -36.00, 'std': 235.17},
        'V3':  {'mean': -63.08, 'median': -50.00, 'std': 261.47},
        'V4':  {'mean': -70.74, 'median': -45.00, 'std': 249.94},
        'V5':  {'mean': -68.42, 'median': -36.00, 'std': 247.15},
        'V6':  {'mean': -57.80, 'median': -29.00, 'std': 316.88},
    }
}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save upload
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Read & preprocess
        data = pd.read_csv(file_path)
        processed = preprocess_ecg_data(data)

        # Basic stats per channel (per column)
        channel_stats = {}
        for col in data.columns:
            channel_stats[col] = {
                'mean':   data[col].mean(),
                'median': data[col].median(),
                'std':    data[col].std(),
                'min':    data[col].min(),
                'max':    data[col].max()
            }

        # Model inference
        orig = data.values
        with torch.no_grad():
            tensor = torch.tensor(processed, dtype=torch.float32)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            output = model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            af_prob = float(probs[0, 1].item())

        diagnosis = "Likely AF" if af_prob > 0.5 else "Likely Normal"

        # Generate interactive plots
        ecg_all_html, ecg_sep_html = generate_ecg_interactive(orig)
        # (correct!)
        stats_html = generate_statistics_plotly(channel_stats, reference_stats)


        # Pass data into template
        return render_template(
            'results.html',
            result={
                'prediction':    af_prob,
                'diagnosis':     diagnosis,
                'statistics':    channel_stats,
                'ecg_all_html':  ecg_all_html,
                'ecg_sep_html':  ecg_sep_html,
                'stats_plots':   stats_html
            }
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
