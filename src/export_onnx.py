import argparse
import os

import tensorflow as tf
import tf2onnx

from src.model.GridTrackNet import GridTrackNet

WIDTH = 768
HEIGHT = 432
IMGS_PER_INSTANCE = 5


def configure_tensorflow():
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def export_onnx(weights_path, onnx_path, opset):
    configure_tensorflow()
    model = GridTrackNet(IMGS_PER_INSTANCE, HEIGHT, WIDTH)
    model.load_weights(weights_path)
    signature = (
        tf.TensorSpec(
            shape=(None, IMGS_PER_INSTANCE * 3, HEIGHT, WIDTH),
            dtype=tf.float32,
            name="input",
        ),
    )
    tf2onnx.convert.from_keras(model, input_signature=signature, opset=opset, output_path=onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GridTrackNet weights to ONNX")
    parser.add_argument("--model_path", required=False, default=os.path.join(os.getcwd(), "model_weights.h5"), type=str, help="Path to TensorFlow weights file.")
    parser.add_argument("--output_path", required=False, default=os.path.join(os.getcwd(), "model_weights.onnx"), type=str, help="Output ONNX model path.")
    parser.add_argument("--opset", required=False, default=17, type=int, help="ONNX opset version. Default = 17")
    args = parser.parse_args()

    export_onnx(args.model_path, args.output_path, args.opset)
    print(f"Exported ONNX model to {args.output_path}")
