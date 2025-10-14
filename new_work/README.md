## Attention-based ODIR Classifier (EfficientNet + Transformer)

This folder contains a stronger baseline than VGG19 by combining a convolutional backbone (EfficientNet) with a lightweight Transformer encoder head for token-wise self-attention on spatial features.

- Backbone: EfficientNetB0/B3 (pretrained ImageNet, trainable)
- Head: Transformer encoder layers over flattened spatial tokens with learnable positional embeddings
- Pooling: Token pooling via global average of Transformer outputs
- Loss: Categorical crossentropy; Metrics: Categorical accuracy, weighted F1 (TensorFlow Addons), PR-AUC/ROC-AUC
- Callbacks: ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

### Data
The notebook expects preprocessed NumPy arrays hosted on Kaggle under `/kaggle/input/...`:

- `x_train.npy`, `y_train.npy`
- `x_val.npy`, `y_val.npy`
- `x_test.npy`, `y_test.npy`

Use any compatible Kaggle dataset path by setting the `DATA_DIR` variable at the top of the notebook (do not use local paths).

### Usage (Kaggle Notebook)
1. Open the notebook `attention_efficientnet_transformer.ipynb` on Kaggle.
2. Set `DATA_DIR` (e.g., `/kaggle/input/5-class-train-test-val`).
3. Run all cells to train. The best model and plots will be saved into `/kaggle/working/`.

### Notes
- Works for 5-class or 7-class settings; the number of classes is inferred from `y_train.shape[1]`.
- If TF Addons is unavailable in your runtime, you can disable the F1 metric by setting `USE_TFA=False` in the notebook.


