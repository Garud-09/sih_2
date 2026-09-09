"""
Depthwise Separable CNN (DS-CNN) Training & INT8 Quantization Pipeline
Targeted for ESP32-S3 TFLite Micro Deployment (< 50 KB Flash, < 48 KB Tensor Arena)

Architecture:
- Input: (49 frames, 40 mel bands, 1 channel)
- Conv2D (16 filters, 3x3, stride 2) + BatchNorm + ReLU
- DepthwiseConv2D (3x3, stride 1) + Conv2D 1x1 (Pointwise, 32 filters) + BatchNorm + ReLU
- DepthwiseConv2D (3x3, stride 1) + Conv2D 1x1 (Pointwise, 48 filters) + BatchNorm + ReLU
- GlobalAveragePooling2D
- Dropout(0.2)
- Dense(2, activation='softmax')
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from dataset_generator import build_dataset
from export_tflite import convert_to_c_header


def build_dscnn_model(input_shape=(49, 40, 1), num_classes=2):
    """Constructs an ultra-compact Depthwise Separable CNN for microcontroller KWS."""
    inputs = keras.Input(shape=input_shape, name="input_spectrogram")

    # Initial 2D Convolution with stride to reduce spatial dimensions
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # DS-CNN Block 1 (Depthwise 3x3 + Pointwise 1x1)
    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (1, 1), strides=(1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # DS-CNN Block 2 (Depthwise 3x3 + Pointwise 1x1)
    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(48, (1, 1), strides=(1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="kws_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="dscnn_kws_isro")
    return model


def quantize_model_int8(model, representative_data, output_path="isro_kws_int8.tflite"):
    """
    Performs full integer post-training quantization (INT8) using representative dataset.
    Generates TFLite model compatible with TensorFlow Lite Micro.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        # Yield representative spectrogram samples
        for i in range(min(150, len(representative_data))):
            sample = np.expand_dims(representative_data[i], axis=0).astype(np.float32)
            yield [sample]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_quant_model)

    size_kb = len(tflite_quant_model) / 1024.0
    print(f"INT8 Quantized Model saved to '{output_path}' ({size_kb:.2f} KB)")
    return output_path, size_kb


def train_and_export(epochs=25, batch_size=32, model_header_path="firmware/arduino/esp32_kws_streamer/model_data.h"):
    print("=== Step 1: Generating Augmented Training & Validation Data ===")
    X, y = build_dataset(num_positive=700, num_negative=700, num_silence=200)
    X = np.expand_dims(X, -1)  # Add channel dimension: (N, 49, 40, 1)

    # Split train / validation (85% / 15%)
    split_idx = int(0.85 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

    print("\n=== Step 2: Compiling Depthwise Separable CNN Model ===")
    model = build_dscnn_model(input_shape=(49, 40, 1), num_classes=2)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    print(f"\n=== Step 3: Training Model for {epochs} Epochs ===")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Accuracy: {val_acc * 100:.2f}%, Loss: {val_loss:.4f}")

    print("\n=== Step 4: Applying Full INT8 Post-Training Quantization ===")
    tflite_file, size_kb = quantize_model_int8(model, X_train, output_path="isro_kws_int8.tflite")
    assert size_kb < 50.0, f"Error: Model size ({size_kb} KB) exceeds 50 KB limit!"

    print("\n=== Step 5: Exporting Quantized Model to C Header ===")
    os.makedirs(os.path.dirname(os.path.abspath(model_header_path)), exist_ok=True)
    convert_to_c_header(tflite_file, model_header_path)
    print(f"Successfully exported C array header to: {model_header_path}")
    print("\n[SUCCESS] Phase 1 TinyML Training & Quantization Pipeline Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Quantize KWS Model for ESP32-S3")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train_and_export(epochs=args.epochs, batch_size=args.batch_size)
