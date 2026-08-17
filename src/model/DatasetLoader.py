import os
import string
import cv2
import re


class DatasetLoader:
    # read all images from the directory and return a list of tuples (filename, image)
    def read_images(self, image_path: str):
        images = []

        for file in os.listdir(image_path):
            if file.lower().endswith((".jpg",".jpeg",".png")):
                image = cv2.imread(os.path.join(image_path, file))
                images.append((file, image))
        return images



