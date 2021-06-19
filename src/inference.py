import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import cv2
import s2sphere
import os


IMAGE_DIR = "../geoguessr"


class GeoImage:
    def __init__(self, image, location):
        self.image = image
        self.location = location


dataframe = pd.read_csv("../planet_v2_labelmap.csv")
model = tf.keras.Sequential(hub.KerasLayer("https://tfhub.dev/google/planet/vision/classifier/planet_v2/1"))


if __name__ == "__main__":

    image_count = 0
    print(len(os.listdir(IMAGE_DIR)))

    while True:
        if len(os.listdir(IMAGE_DIR)) != image_count:
            image_count = len(os.listdir(IMAGE_DIR))

            geo_images = []
            for filename in os.listdir(IMAGE_DIR):
                image = np.array(cv2.resize(cv2.imread(os.path.join(IMAGE_DIR, filename)), (299, 299)) / 255.0)
                geo_images.append(GeoImage(image, filename))
            geo_images = np.array(geo_images)
            images = [geo_image.image for geo_image in geo_images]

            for geo_image in geo_images:
                predictions = model.predict(geo_image.image[np.newaxis, ...])
                geo_guess = np.argmax(predictions)
                cell_token = dataframe[dataframe["id"] == geo_guess].iloc[0]["S2CellId"]
                point = s2sphere.CellId().from_token(cell_token).to_lat_lng()

                print("{}: {}".format(geo_image.location, point))





