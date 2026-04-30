**Industrial Sound Anomaly Detection**

This project detects abnormal fan sounds using two methods:

1. **Method 1: Autoencoder anomaly detection**
2. **Method 2: Supervised CNN classification**

**Overview**

Audio files are split into 1-second chunks, converted into log-mel spectrograms, normalized, and evaluated at the file level using the top-k chunk scores.

**Installation**

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

**Method 1: Autoencoder**

- Trains only on normal audio.
- Uses reconstruction error and latent-space distance.
- Tunes reconstruction/latent score weights on validation files.
- Uses a threshold learned from normal training scores.

Run:

```powershell
python main_autoencoder.py
```

Run with the included sample dataset:

```powershell
python main_autoencoder.py --dataset-path sample_dataset\fan\id_00
```

Output plot:

```text
method1_autoencoder_file_level_scores.png
```

**Method 2: Supervised CNN**

- Trains on both normal and abnormal labels.
- Predicts abnormal probability for each chunk.
- Uses the average of the top-k chunk probabilities as the file score.
- This method gives the strongest results in the current experiments.

Run:

```powershell
python main_classifier.py
```

Run with the included sample dataset:

```powershell
python main_classifier.py --dataset-path sample_dataset\fan\id_00
```

Output plot:

```text
method2_supervised_cnn_file_probabilities.png
```

**Compare Methods**

Compare both methods on the configured dataset ID:

```powershell
python compare_methods.py
```

Compare both methods with the included sample dataset:

```powershell
python compare_methods.py --dataset-path sample_dataset\fan\id_00
```

Compare both methods across all `dataset/fan/id_*` folders:

```powershell
python compare_all_ids.py
```

Compare all IDs under a custom dataset root:

```powershell
python compare_all_ids.py --dataset-root sample_dataset\fan
```

**Full Dataset Commands**

After downloading and arranging the complete Zenodo dataset under `dataset/fan`, run one ID like this:

```powershell
python main_autoencoder.py --dataset-path dataset\fan\id_00
python main_classifier.py --dataset-path dataset\fan\id_00
python compare_methods.py --dataset-path dataset\fan\id_00
```

Run all available IDs:

```powershell
python compare_all_ids.py --dataset-root dataset\fan
```

If the full dataset uses the default path in `src/config.py`, this also works:

```powershell
python compare_methods.py
```

Outputs:

```text
all_ids_method_comparison.csv
methods_autoencoder_vs_supervised_cnn_comparison_id_00.png
methods_autoencoder_vs_supervised_cnn_comparison_id_02.png
methods_autoencoder_vs_supervised_cnn_comparison_id_04.png
methods_autoencoder_vs_supervised_cnn_comparison_id_06.png
```

**Project Structure**

```text
src/
  config.py
  dataset.py
  features.py
  model.py
  train_autoencoder.py
  evaluate_autoencoder.py
  tuning_autoencoder.py
  train_classifier.py
  evaluate_classifier.py

main_autoencoder.py
main_classifier.py
compare_methods.py
compare_all_ids.py
README.md
```

**Dataset Structure**

The experiments use the public dataset from Zenodo:

```text
https://zenodo.org/records/3384388
```

The full dataset is not uploaded to this repository because it is about 10 GB.

A small sample subset is included for structure inspection and quick loading tests. It contains 20 normal WAV files and 20 abnormal WAV files from `id_00`:

```text
sample_dataset/
  fan/
    id_00/
      normal/
      abnormal/
```

The sample subset is still small for meaningful model training. Use the full Zenodo dataset for real experiments.

Expected structure:

```text
dataset/
  fan/
    id_00/
      normal/
      abnormal/
    id_02/
      normal/
      abnormal/
```

Additional IDs such as `id_04` and `id_06` are automatically detected by `compare_all_ids.py`.

**Current Insight**

The tuned autoencoder improves recall, but it still produces many false positives. The supervised CNN performs much better across all tested IDs while adding very little inference time compared with the autoencoder.

**Embedded Deployment Notes**

The current repository focuses on Python training, evaluation, benchmarking, and method comparison. It is not yet a complete STM32 firmware project.

Potential STM32 deployment paths:

- Export the trained CNN to ONNX or another intermediate format.
- Convert/deploy using STM32Cube.AI inside STM32CubeIDE when possible.
- Use TensorFlow Lite Micro only if the target workflow supports it cleanly.
- Port the log-mel feature extraction to C/CMSIS-DSP or another optimized embedded DSP implementation.
- Apply quantization to reduce memory and inference cost.

Benchmarking shows that model inference is small compared with feature extraction, so embedded optimization should focus first on the log-mel pipeline.

**Future Work**

- Improve detection accuracy using more advanced scoring methods.
- Add temporal smoothing for more stable predictions.
- Enable real-time audio processing.
- Deploy on embedded systems such as STM32.
- Apply model optimization techniques such as quantization.
