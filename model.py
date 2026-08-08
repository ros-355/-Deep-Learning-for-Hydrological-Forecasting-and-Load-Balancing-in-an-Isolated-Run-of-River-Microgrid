"""Compact LSTM, selective QAT, and full-integer TensorFlow Lite export."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from .config import ModelConfig


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm_model(
    feature_count: int,
    config: ModelConfig | None = None,
    batch_size: int | None = None,
) -> keras.Model:
    """Build a compact multi-output LSTM for direct 24-hour forecasting."""

    cfg = config or ModelConfig()
    set_reproducible_seed(cfg.random_seed)
    if batch_size is None:
        inputs = keras.Input(
            shape=(cfg.lookback_hours, feature_count), name="history_72h"
        )
    else:
        inputs = keras.Input(
            batch_shape=(batch_size, cfg.lookback_hours, feature_count),
            name="history_72h",
        )
    x = keras.layers.LSTM(
        cfg.lstm_units,
        activation="tanh",
        recurrent_activation="sigmoid",
        dropout=0.0,
        recurrent_dropout=0.0,
        unroll=False,
        name="temporal_encoder",
    )(inputs)
    x = keras.layers.Dense(cfg.dense_units, activation="relu", name="projection")(x)
    residual = keras.layers.Dense(
        cfg.forecast_hours,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="flow_residual_24h",
    )(x)
    last_observed_flow = inputs[:, -1, 0:1]
    outputs = keras.layers.Add(name="flow_forecast_24h")(
        [residual, last_observed_flow]
    )
    model = keras.Model(inputs=inputs, outputs=outputs, name="tiny_hydro_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_float_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: ModelConfig | None = None,
    verbose: int = 2,
) -> keras.callbacks.History:
    cfg = config or ModelConfig()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1.0e-5
        ),
    ]
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
    )


def build_qat_model(
    trained_model: keras.Model,
    config: ModelConfig | None = None,
) -> keras.Model:
    """Apply TFMOT fake quantization to supported Dense layers and fine-tune.

    TFMOT's default 8-bit registry does not cover recurrent LSTM layers. This
    deployment-safe implementation follows the documented selective-QAT path:
    both Dense layers receive fake-quant nodes during fine-tuning, while the
    subsequent representative-data TFLite conversion quantizes the complete
    graph, including the fused recurrent operator, to integer kernels.
    """

    cfg = config or ModelConfig()

    def annotate_dense(layer: keras.layers.Layer) -> keras.layers.Layer:
        if isinstance(layer, keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    annotated = keras.models.clone_model(trained_model, clone_function=annotate_dense)
    annotated.set_weights(trained_model.get_weights())
    qat_model = tfmot.quantization.keras.quantize_apply(annotated)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.qat_learning_rate),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return qat_model


def build_fixed_batch_qat_clone(
    trained_qat_model: keras.Model,
    feature_count: int,
    config: ModelConfig | None = None,
) -> keras.Model:
    """Clone trained QAT weights into a batch-one graph for fused TFLite export."""

    cfg = config or ModelConfig()
    fixed_float_model = build_lstm_model(feature_count, cfg, batch_size=1)
    fixed_qat_model = build_qat_model(fixed_float_model, cfg)
    fixed_qat_model.set_weights(trained_qat_model.get_weights())
    return fixed_qat_model


def fine_tune_qat(
    qat_model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: ModelConfig | None = None,
    verbose: int = 2,
) -> keras.callbacks.History:
    cfg = config or ModelConfig()
    return qat_model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=cfg.qat_epochs,
        batch_size=cfg.batch_size,
        shuffle=False,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ],
        verbose=verbose,
    )


def _representative_dataset(
    samples: np.ndarray, max_samples: int = 256
) -> Callable[[], Iterable[list[np.ndarray]]]:
    stride = max(len(samples) // max_samples, 1)

    def generator() -> Iterable[list[np.ndarray]]:
        count = 0
        for index in range(0, len(samples), stride):
            yield [samples[index : index + 1].astype(np.float32)]
            count += 1
            if count >= max_samples:
                break

    return generator


def export_full_int8_tflite(
    qat_model: keras.Model,
    representative_samples: np.ndarray,
    output_path: str | Path,
) -> bytes:
    """Export an INT8-input/INT8-output flatbuffer and validate its signature."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset(representative_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()
    output.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_dtype = interpreter.get_input_details()[0]["dtype"]
    output_dtype = interpreter.get_output_details()[0]["dtype"]
    if input_dtype != np.int8 or output_dtype != np.int8:
        raise RuntimeError(
            f"Expected INT8 I/O, received input={input_dtype}, output={output_dtype}."
        )
    return tflite_model


def tflite_memory_report(tflite_model: bytes) -> dict[str, float | str]:
    """Report exact flatbuffer size and a conservative live-tensor upper bound."""

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    tensor_bytes = 0
    for detail in interpreter.get_tensor_details():
        shape = np.asarray(detail["shape"], dtype=np.int64)
        element_count = int(np.prod(shape)) if shape.size else 1
        tensor_bytes += element_count * np.dtype(detail["dtype"]).itemsize
    flatbuffer_kb = len(tflite_model) / 1024.0
    tensor_upper_bound_kb = tensor_bytes / 1024.0
    combined_upper_bound_kb = flatbuffer_kb + tensor_upper_bound_kb
    return {
        "flatbuffer_kb": float(flatbuffer_kb),
        "tensor_storage_upper_bound_kb": float(tensor_upper_bound_kb),
        "combined_upper_bound_kb": float(combined_upper_bound_kb),
        "below_500_kb": bool(combined_upper_bound_kb < 500.0),
        "note": (
            "Tensor storage is a conservative sum, not a board-specific TFLite "
            "Micro arena measurement; validate the arena on the target firmware."
        ),
    }


def predict_tflite_int8(tflite_model: bytes, x: np.ndarray) -> np.ndarray:
    """Run quantized inference while respecting TFLite scale and zero point."""

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    if input_scale == 0.0 or output_scale == 0.0:
        raise RuntimeError("TFLite quantization parameters are missing.")

    predictions = np.empty((len(x), *output_detail["shape"][1:]), dtype=np.float32)
    for index, sample in enumerate(x):
        quantized_input = np.clip(
            np.round(sample / input_scale + input_zero), -128, 127
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized_input[None, ...])
        interpreter.invoke()
        quantized_output = interpreter.get_tensor(output_detail["index"])[0]
        predictions[index] = (quantized_output.astype(np.float32) - output_zero) * output_scale
    return predictions


def print_memory_report(report: dict[str, float | str]) -> None:
    print("\nTinyML memory report")
    print(f"  INT8 TFLite flatbuffer: {report['flatbuffer_kb']:.2f} KB")
    print(
        "  Conservative tensor-storage upper bound: "
        f"{report['tensor_storage_upper_bound_kb']:.2f} KB"
    )
    print(f"  Combined upper bound: {report['combined_upper_bound_kb']:.2f} KB")
    print(f"  Below 500 KB threshold: {report['below_500_kb']}")
    print(f"  Qualification: {report['note']}")
