# Shoplifting Detection: Video Classification

Binary video classification (normal shopping behavior vs. shoplifting) with temporal deep learning models trained from scratch in PyTorch. Built for the Cellula Technologies computer vision task.

Three architectures are implemented, each in its own notebook, on top of the same data pipeline:

| Notebook | Model | Parameters |
|---|---|---|
| `notebooks/cnn_lstm.ipynb` | Residual CNN + BiLSTM, attention pooling, appearance shortcut | 3.27 M |
| `notebooks/cnn_transformer.ipynb` | Residual CNN + Transformer encoder, attention pooling, residual shortcut | 3.41 M |
| `notebooks/cnn3d.ipynb` | (2+1)D residual 3D CNN | 3.52 M |

No pretrained weights are used anywhere.

## Dataset

855 surveillance clips in two classes, loaded from a zip file (the link is set in `DATA_URL` in the first cell of each notebook).

| | Normal | Shoplifting | Total |
|---|---|---|---|
| Clips | 531 | 324 | 855 |

- Resolution: 704 x 576 for every clip
- Frame rate: about 25 fps
- Duration: 3 to 74 s, median 13 s (75 to 1850 frames)

Class labels are inferred from the folder names (`non shop lifters` and `shop lifters`).

## Pipeline

All three notebooks share these steps.

1. **Frame sampling.** `sample_indices(n, t)` computes a uniform skip interval `n / t` and takes one frame per segment, so the sampled frames span the whole clip for any length. It is verified on clips of 1 to 5000 frames. During training each frame is jittered inside its segment.
2. **Frame cache.** Each video is decoded once into 32 uniformly spaced frames at 128 x 128 and kept in memory. Training then reads from the cache and never decodes video.
3. **EDA.** Class balance, duration, frame count and resolution; motion, brightness and contrast per class; sample frames; a metadata-only baseline.
4. **Split.** 70% train, 15% validation, 15% test, stratified by class. See "Evaluation notes" for the duplicate handling.
5. **Augmentation.** Horizontal flip, random resized crop and color jitter. Every random parameter is drawn once per clip and applied to all frames. A code check confirms that all frames of a clip come out identical.
6. **Dataset class.** Returns a tensor of shape `(T, C, H, W)` = `(16, 3, 112, 112)` per clip, plus the label.
7. **Input to the models.** RGB plus a motion channel (the difference to the previous sampled frame), built inside the model.
8. **Training.** AdamW, learning rate 1e-3 with 3 epochs of warmup and cosine decay, weighted binary cross-entropy, gradient clipping, mixed precision, and an exponential moving average (EMA) of the weights. At most 80 epochs, early stopping with patience 25. The best epoch is chosen on validation AUC and F1.
9. **Evaluation.** Accuracy, precision, recall, F1 and AUC on the test set, with flip test-time augmentation. The decision threshold is reported at 0.5 and at the value that maximizes validation F1.

A short sanity cell before training checks that the model can memorize 32 clips, which catches pipeline bugs early.

## Models

### CNN + LSTM
A residual CNN (stem + three stages of two blocks, 64/128/256 channels) encodes each frame into a 256-d vector. A bidirectional LSTM reads the 16 vectors. The classifier receives three things concatenated: attention-pooled LSTM output, max-pooled LSTM output, and the mean of the frame vectors (appearance shortcut).

### CNN + Transformer
The same frame encoder, followed by a one-layer pre-norm Transformer encoder (4 heads, learned positional embedding) with a residual connection around it. The same three-part pooling feeds the classifier.

### 3D CNN
A (2+1)D residual network: each 3D convolution is factorized into a spatial 3x3 and a temporal 3x1 convolution. A (1x5x5) stem keeps all 16 frames; time is halved in the two later stages (16 to 8 to 4). The head uses global average and max pooling.

All models use Kaiming initialization and zero-initialized last BatchNorm scales in the residual blocks, which makes training from scratch stable.

## Results

Test set, threshold 0.5 / threshold tuned on validation:

| Model | Split | Test clips | Accuracy | F1 | AUC |
|---|---|---|---|---|---|
| CNN + LSTM | all 855 clips | 129 | 0.977 / 0.977 | 0.970 / 0.969 | 0.997 |
| 3D CNN | all 855 clips | 129 | 0.961 / 0.969 | 0.952 / 0.961 | 0.991 |
| CNN + Transformer | duplicates removed | 92 | 0.859 / 0.870 | 0.860 / 0.872 | 0.939 |

## Evaluation notes

- **Duplicates.** 237 of the 855 clips are exact copies of other clips (nearly all in the normal class). The Transformer notebook removes them and splits by group, so near-identical scenes stay in one split. The LSTM and 3D CNN notebooks were run before this step, with copies spread across train, validation and test. Their scores are therefore optimistic, and **the three rows above are not directly comparable**. A fair comparison needs the duplicate cells from `cnn_transformer.ipynb` in every notebook.
- **Test set size.** With 92 to 129 test clips, one clip moves accuracy by about 1 point. Differences of 2 points or less between models are within noise.
- **Shared camera scenes.** Clips from the same camera can still appear in different splits. Only exact duplicates and close copies are grouped.
- **Metadata shortcuts.** In the EDA, brightness alone reaches an AUC of 0.64 and a metadata-only logistic regression reaches 0.76, so part of the signal in this dataset is scene appearance and not behavior.
- **Sparse sampling.** 16 frames cover a whole clip (about 0.8 s between frames), so fast hand movements are only coarsely visible.

## How to run

1. Open a notebook on Kaggle (or Colab) with a GPU and internet enabled.
2. Set `DATA_URL` in the first cell to a zip link (Google Drive links work through `gdown`), or leave it empty and attach the data as a Kaggle dataset.
3. Run all cells. Training takes roughly 10 to 25 minutes on a T4 GPU.

Requirements: Python 3.10+, PyTorch, OpenCV, NumPy, pandas, scikit-learn, SciPy, matplotlib, joblib, tqdm, gdown. All are preinstalled on Kaggle.

## Repository layout

```
.
├── README.md
└── notebooks/
    ├── cnn_lstm.ipynb
    ├── cnn_transformer.ipynb
    └── cnn3d.ipynb
```

## References

- J. Donahue et al. *Long-term Recurrent Convolutional Networks for Visual Recognition and Description.* CVPR 2015. arXiv:1411.4389
- D. Tran et al. *A Closer Look at Spatiotemporal Convolutions for Action Recognition* (R(2+1)D). CVPR 2018. arXiv:1711.11248
- K. He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- P. Goyal et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* 2017. arXiv:1706.02677 (zero-initialized residual BatchNorm)
- I. Muneer, M. Saddique, Z. Habib, H. G. Mohamed. *Shoplifting Detection Using Hybrid Neural Network CNN-BiLSTM and Development of Benchmark Dataset.* Applied Sciences 13(14), 8341, 2023. doi:10.3390/app13148341
- M. A. Ansari, D. K. Singh. *Optimized Parameter Tuning in a Recurrent Learning Process for Shoplifting Activity Classification.* Cybernetics and Information Technologies 23(1), 2023. doi:10.2478/cait-2023-0008
