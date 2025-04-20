# ECG Atrial Fibrillation (AF) Detector

A web-based application that uses deep learning to detect Atrial Fibrillation from ECG signals. This application processes uploaded ECG data and provides real-time analysis using a trained Stacked LSTM with multi-pooling architecture.

## Features

- **ECG Signal Upload**: Upload CSV files containing 12-lead ECG data
- **Real-time Prediction**: Get instant AF detection results
- **Statistical Analysis**: View comprehensive statistics about the uploaded ECG data
- **Visualization**: Interactive plots showing ECG signals and comparative statistics
- **User-friendly Interface**: Simple web interface for easy ECG analysis

## Project Structure

```
ecg_af_detector/
├── app.py                  # Flask application
├── requirements.txt        # Dependencies
├── static/                 # Static files
│   ├── css/
│   │   └── style.css       # Custom styling
│   ├── js/
│   │   └── main.js         # Frontend JavaScript
│   └── images/             # Images for the website
├── templates/              # HTML templates
│   ├── index.html          # Main page
│   └── results.html        # Results page
├── models/                 # Store the trained model
│   └── ecg_model.pth       # Trained PyTorch model
│   └── model_architecture.py # Model architecture definition
└── utils/                  # Utility functions
    ├── __init__.py
    ├── data_processing.py  # Data processing functions
    └── visualization.py    # Visualization functions
```

## Requirements

- Python 3.8+
- PyTorch
- Flask
- NumPy
- Pandas
- Matplotlib
- SciPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ecg-af-detector.git
cd ecg-af-detector
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure your trained model is placed in the `models/` directory as `ecg_model.pth`

## Running the Application

1. Start the Flask development server:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. **Upload ECG Data**: Click on the upload button and select your ECG CSV file. The file should contain 12-lead ECG data with dimensions (5000, 12) where 5000 represents time points and 12 represents the ECG leads.

2. **View Results**: After processing, you'll see:
   - Prediction probability for AF
   - Diagnosis (Likely AF or Likely Normal)
   - Statistical analysis of the ECG signal
   - ECG signal visualization
   - Comparative statistical plots

## Model Architecture

The application uses a Stacked LSTM with multi-pooling architecture that processes ECG signals through multiple LSTM layers with attention pooling mechanisms. The model expects input with shape (batch_size, 12, 5000) representing 12-lead ECG data with 5000 time points per lead.

## Data Format

The input CSV file should have:
- 5000 rows (time points)
- 12 columns (ECG leads)
- Numeric values representing ECG signal amplitude

The application will automatically handle necessary preprocessing including normalization and dimension adjustment.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Stacked LSTM architecture is based on state-of-the-art ECG classification research
- Thanks to all contributors who have helped with this project
