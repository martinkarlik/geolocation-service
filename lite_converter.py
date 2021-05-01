import tensorflow as tf

MODEL_PATH = "models/planet.h5"


def convert(path):

    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(path))
    tflite_model = converter.convert()

    with open('models/planet.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    convert(MODEL_PATH)
