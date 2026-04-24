**Industrial Sound Anomaly Detection**

This project implements an end-to-end anomaly detection system for industrial machine sounds, with a focus on fan noise.

**Overview**

The system detects abnormal behavior in audio signals by learning patterns of normal operation and identifying deviations. It uses a deep learning approach based on a CNN autoencoder and latent-space analysis.

**Methodology**

The pipeline consists of:

1. **Audio Processing**  
- Convert raw audio into log-mel spectrograms  
- Normalize features using training data statistics  

2. **Model**  
- CNN-based autoencoder  
- Learns compressed representations of normal sound patterns  

3. **Anomaly Detection**  
- Extract latent representation (encoder output)  
- Compute distance from learned normal latent center  
- Use thresholding to classify anomalies  

4. **Evaluation**  
- Metrics:
  - False Positive Rate (FPR)  
  - True Positive Rate (Recall)  
- Visualization using histograms  

**Example Results**

- TPR (Recall): approximately 30–50%  
- FPR: approximately 10–25%  

These results demonstrate the trade-off between detection sensitivity and false alarms.

**How to Run**

Install dependencies:
pip install -r requirements.txt

Run the pipeline:
python main.py

**Project Structure**

anomalyDetctionInSound/

src/  
- dataset.py       (data loading and preprocessing)  
- features.py      (feature extraction using log-mel spectrograms)  
- model.py         (CNN autoencoder model)  
- train.py         (training loop)  
- evaluate.py      (anomaly scoring and metrics)  
- config.py        (configuration)  

main.py            (entry point)  
README.md  

**Dataset**

The dataset is not included in this repository due to size constraints.

Please place your dataset in the following structure:

dataset/  
  train/  
  test/  

**Key Features**

- Latent-space anomaly detection (stronger than reconstruction error)  
- Threshold tuning for balancing false positives and recall  
- Modular and extensible pipeline  
- Designed with potential for embedded deployment (e.g., STM32 using TensorFlow Lite Micro)  

**Future Work**

- Improve detection accuracy using advanced scoring methods  
- Add temporal smoothing for more stable predictions  
- Enable real-time audio processing  
- Deploy on embedded systems (STM32)  
- Apply model optimization techniques such as quantization  

**Key Insight**

A key finding in this project is that stronger models can negatively impact anomaly detection by reconstructing abnormal patterns too well. Using latent-space distance provides better separation between normal and anomalous behavior.
