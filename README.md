# Shoplifting Detection: Video Classification

Binary video classification of surveillance clips into **normal shopping behavior** and **shoplifting** using temporal deep learning models implemented in PyTorch. This project was developed for the Cellula Technologies computer vision task.

The repository explores three architectures using a shared video-processing and evaluation workflow:

| Notebook | Model | Parameters |
|---|---|---:|
| `notebooks/cnn_lstm.ipynb` | Residual CNN + BiLSTM with attention pooling and an appearance shortcut | 3.27 M |
| `notebooks/cnn_transformer.ipynb` | Residual CNN + Transformer encoder | 3.41 M |
| `notebooks/cnn3d.ipynb` | (2+1)D residual 3D CNN | 3.52 M |

The LSTM notebook has been updated with duplicate/near-duplicate analysis and group-aware data splitting. The classifier is trained from scratch; a pretrained ResNet-50 is used only to create frame embeddings for the duplicate/group analysis.

## Dataset

The notebooks use a dataset of surveillance clips organized into two classes:

- **Normal:** `non shop lifters`
- **Shoplifting:** `shop lifters`

The dataset is downloaded from the URL configured in `DATA_URL` in the notebook, or can be attached directly as a Kaggle dataset. The original dataset scan found 855 clips (531 normal and 324 shoplifting). Dataset counts can differ after filtering or when using a different copy of the data.

Video metadata in the original scan:

- Resolution: 704 × 576
- Frame rate: approximately 25 fps
- Duration: 3–74 seconds, with a median of approximately 13 seconds

## Pipeline

The updated CNN + LSTM notebook includes the following workflow:

1. **Video loading and label inference.** Video paths and class labels are inferred from folder/file names.
2. **Frame sampling.** Clips are represented by uniformly sampled frames across the full video. Training applies temporal jitter to the sampled indices.
3. **Frame cache and metadata.** Frames are decoded and resized once for reuse. Video duration, frame count, frame rate, resolution, motion, brightness, and contrast are explored.
4. **Exploratory data analysis.** Class distribution, clip metadata, representative frames, and a metadata-only baseline are examined.
5. **Duplicate and similarity analysis.** A pretrained ResNet-50 generates frame embeddings to identify visually similar clips. Duplicate and grouping thresholds are inspected before splitting.
6. **Group-aware split.** `StratifiedGroupKFold` is used to create approximate train/validation/test splits while keeping similar clips in the same split to reduce data leakage.
7. **Augmentation.** Training uses horizontal flips, random crops, and color/contrast adjustments. The same random spatial transformation is applied consistently across all frames in a clip.
8. **Model input.** The dataset returns 16 RGB frames per clip, resized to 112 × 112. The model additionally constructs a motion input from differences between consecutive frames.
9. **Training.** The CNN + BiLSTM is trained from scratch using AdamW, weighted binary cross-entropy, learning-rate warmup and cosine decay, gradient clipping, mixed precision when CUDA is available, exponential moving average (EMA), and early stopping.
10. **Evaluation.** Accuracy, precision, recall, F1-score, and ROC-AUC are measured on the test split. Results are reported using both a 0.5 threshold and a threshold selected using validation F1. Horizontal-flip test-time augmentation is used.

A sanity check before full training verifies that the model can overfit a small subset of clips, helping catch basic pipeline or training issues.

## Models

### CNN + LSTM

A residual CNN encodes each frame into a feature vector. The sequence of frame features is processed by a bidirectional LSTM. Attention pooling and max pooling summarize the temporal features, while a mean-pooled appearance shortcut preserves clip-level visual information. The combined representation is passed to a binary classifier.

The network takes RGB frames and a motion channel derived from frame-to-frame differences. Its convolutional classifier is initialized and trained from scratch.

### CNN + Transformer

A residual CNN extracts frame-level features, which are processed by a Transformer encoder with positional information. The temporal representation is pooled and passed to a binary classifier.

### 3D CNN

A (2+1)D residual network factorizes 3D convolutions into spatial and temporal operations to learn appearance and motion patterns jointly.

## Results

### Updated CNN + LSTM notebook

The latest saved evaluation output reports the following test metrics:

| Decision threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 0.50 | 0.8780 | 0.9348 | 0.8600 | 0.8958 | 0.9681 |
| 0.27 (selected on validation) | 0.8902 | 0.8596 | 0.9800 | 0.9159 | 0.9681 |

The threshold selected on the validation set increases recall and F1-score on the saved test evaluation. The 0.5 threshold gives higher precision. The reported results are from the notebook's saved outputs and may change when the notebook is rerun with different data, environment, or training conditions.

### Comparison with earlier model runs

The Transformer and 3D CNN results below are from earlier runs and should not be treated as a controlled comparison with the updated LSTM run. Their split and duplicate-handling procedures differed.

| Model | Evaluation split | Test clips | Accuracy | F1-score | ROC-AUC |
|---|---|---:|---:|---:|---:|
| CNN + Transformer | Duplicate-aware split | 92 | 0.859–0.870 | 0.860–0.872 | 0.939 |
| 3D CNN | Earlier split | 129 | 0.961–0.969 | 0.952–0.961 | 0.991 |

The earlier CNN + LSTM result is intentionally not repeated here because the notebook has since been updated with a different split procedure. For a fair model comparison, all architectures should use the same duplicate-aware grouping and train/validation/test split.

## Evaluation considerations

- **Potential data leakage:** Similar clips can inflate test scores if related videos appear across different splits. The updated LSTM notebook uses visual similarity grouping to reduce this risk.
- **Threshold selection:** The 0.5 threshold and the validation-selected threshold reflect different precision/recall trade-offs. The threshold must be selected using validation data, not the test set.
- **Limited test size:** Metrics can vary noticeably when the test set contains a relatively small number of clips.
- **Scene-related shortcuts:** Brightness, background, and other visual characteristics may correlate with labels. High scores do not necessarily mean the model has learned shoplifting behavior alone.
- **Sparse temporal sampling:** Sampling a limited number of frames across a clip can miss brief actions or fast hand movements.

## How to run

1. Open the notebook in Kaggle or Google Colab with internet access and, ideally, a GPU.
2. Set `DATA_URL` in the first code cell to a downloadable ZIP link, or set it up to use a dataset attached to the notebook.
3. Run the notebook cells in order.

Training is configured for up to 80 epochs with early stopping. Runtime depends on the available hardware and dataset.

## Requirements

Python 3.10+, PyTorch, OpenCV, NumPy, pandas, scikit-learn, SciPy, matplotlib, joblib, tqdm, and `gdown`. The notebook also uses `torchvision` for the ResNet-50 embeddings in duplicate analysis. Most dependencies are available by default in Kaggle notebooks.

## Repository layout

```text
.
├── README.md
└── notebooks/
    ├── cnn_lstm.ipynb
    ├── cnn_transformer.ipynb
    └── cnn3d.ipynb
```

Update the notebook paths in this section if your repository uses different filenames or locations.

## References

- J. Donahue et al. *Long-term Recurrent Convolutional Networks for Visual Recognition and Description.* CVPR 2015. arXiv:1411.4389
- D. Tran et al. *A Closer Look at Spatiotemporal Convolutions for Action Recognition* (R(2+1)D). CVPR 2018. arXiv:1711.11248
- K. He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- P. Goyal et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* 2017. arXiv:1706.02677
- I. Muneer, M. Saddique, Z. Habib, H. G. Mohamed. *Shoplifting Detection Using Hybrid Neural Network CNN-BiLSTM and Development of Benchmark Dataset.* Applied Sciences 13(14), 8341, 2023. doi:10.3390/app13148341
- M. A. Ansari, D. K. Singh. *Optimized Parameter Tuning in a Recurrent Learning Process for Shoplifting Activity Classification.* Cybernetics and Information Technologies 23(1), 2023. doi:10.2478/cait-2023-0008

