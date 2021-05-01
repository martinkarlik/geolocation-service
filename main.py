import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.compat.v1
import numpy as np
import pandas as pd
import cv2
import s2sphere
import os

IMAGE_DIR = "custom_images"


class GeoImage:
    def __init__(self, image, location):
        self.image = image
        self.location = location


# tensorflow.compat.v1.disable_eager_execution()
dataframe = pd.read_csv("planet_v2_labelmap.csv")


geo_images = []
for filename in os.listdir(IMAGE_DIR):
    location = filename.split('_')[0]
    if not location.isdigit():
        image = np.array(cv2.resize(cv2.imread(os.path.join(IMAGE_DIR, filename)), (299, 299)) / 255.0)
        geo_images.append(GeoImage(image, location))
geo_images = np.array(geo_images)

# module = hub.Module("https://tfhub.dev/google/planet/vision/classifier/planet_v2/1")
# height, width = hub.get_expected_image_size(module)
# features = module(images)


model = tf.keras.Sequential(hub.KerasLayer("https://tfhub.dev/google/planet/vision/classifier/planet_v2/1"))


for geo_image in geo_images:
    predictions = model.predict(geo_image.image[np.newaxis, ...])
    geo_guess = np.argmax(predictions)
    cell_token = dataframe[dataframe["id"] == geo_guess].iloc[0]["S2CellId"]

    point = s2sphere.CellId().from_token(cell_token).to_lat_lng()

    print("{}: {}".format(geo_image.location, point))
    cv2.imshow(geo_image.location, geo_image.image)

    cv2.waitKey()

cv2.destroyAllWindows()




# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.experimental_new_converter = True
# tflite_model = converter.convert()
#
# with open('models/planet.tflite', 'wb') as f:
#     f.write(tflite_model)

