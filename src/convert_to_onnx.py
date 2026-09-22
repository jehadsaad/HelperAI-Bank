import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model(
    "models/helperai_bank_model.keras"
)

spec = (
    tf.TensorSpec(
        (None, 23),
        tf.float32,
        name="input"
    ),
)

tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path="models/helperai_bank_model.onnx"
)

print("ONNX model converted successfully!")