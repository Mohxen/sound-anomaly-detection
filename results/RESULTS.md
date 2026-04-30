**Results**

These results compare two methods across the available fan IDs:

1. **Autoencoder anomaly detection**
2. **Supervised CNN classifier**

The supervised CNN consistently performs better, with higher recall and much lower false positive rate.

**All-ID Summary**

| Dataset ID | Method | Recall | False Positive Rate | F1 |
|---|---|---:|---:|---:|
| id_00 | Autoencoder | 0.6601 | 0.3861 | 0.7128 |
| id_00 | Supervised CNN | 0.9383 | 0.0050 | 0.9620 |
| id_02 | Autoencoder | 0.8436 | 0.5149 | 0.7906 |
| id_02 | Supervised CNN | 1.0000 | 0.0148 | 0.9796 |
| id_04 | Autoencoder | 0.7299 | 0.4466 | 0.7320 |
| id_04 | Supervised CNN | 1.0000 | 0.0290 | 0.9589 |
| id_06 | Autoencoder | 0.9834 | 0.4455 | 0.8812 |
| id_06 | Supervised CNN | 1.0000 | 0.0049 | 0.9931 |

**Takeaway**

The tuned autoencoder improves recall, but it produces many false positives. The supervised CNN is the stronger method for this dataset because it achieves high recall with a much lower false positive rate.

**Generated Files**

```text
results/all_ids_method_comparison.csv
results/methods_autoencoder_vs_supervised_cnn_comparison_id_00.png
results/methods_autoencoder_vs_supervised_cnn_comparison_id_02.png
results/methods_autoencoder_vs_supervised_cnn_comparison_id_04.png
results/methods_autoencoder_vs_supervised_cnn_comparison_id_06.png
```
