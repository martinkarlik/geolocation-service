import tflit
import os
import cv2
import s2sphere
import numpy as np
import pandas as pd

IMAGE_DIR = "../custom_images"
MODEL_PATH = "../models/planet.tflite"


class GeoImage:
    def __init__(self, image, location):
        self.image = image
        self.location = location


dataframe = pd.read_csv("../planet_v2_labelmap.csv")
model = tflit.Model(MODEL_PATH)


geo_images = []
for filename in os.listdir(IMAGE_DIR):
    location = filename.split('_')[0]
    if not location.isdigit() and location == "hasseris.jpg":
        image = np.array(cv2.resize(cv2.imread(os.path.join(IMAGE_DIR, filename)), (299, 299)) / 255.0)

        print(image[0, 0], image[0, 298], image[298, 298], image[298, 0])

        geo_images.append(GeoImage(image, location))
geo_images = np.array(geo_images)


for geo_image in geo_images:

    predictions = model.predict(geo_image.image[np.newaxis, ...])

    geo_guess = np.argmax(predictions)

    print("Geo guess: {}, value: {}".format(geo_guess, predictions[0, geo_guess]))
    cell_token = dataframe[dataframe["id"] == geo_guess].iloc[0]["S2CellId"]

    point = s2sphere.CellId().from_token(cell_token).to_lat_lng()

    print(point)
    cv2.imshow(geo_image.location, geo_image.image)

    cv2.waitKey()

cv2.destroyAllWindows()

model.summary()
