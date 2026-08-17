import cv2;
from model.DatasetLoader import DatasetLoader
import numpy as np
from model.feature_extraction.FeatureExtractorStrategy import FeatureExtractorStrategy


class EdgeHistogram(FeatureExtractorStrategy):

    # initialize the edge histogram
    def __init__(
        self,
        threshold1: int = 100,
        threshold2: int = 200,
        bins: int = 16,
        image_size: tuple = (128, 128),
    ):

        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.bins = bins
        self.image_size = image_size

    def extract(self, image: np.ndarray) -> np.ndarray:
        self.__validate(image)
        gray = self.__grayScale(image)
        edges = self.__edges(gray)
        histogram = self.__histogram(edges)
        normalized = self.__normalize(histogram)
        return normalized

    def __validate(self, image):
        pass
    
    def __grayScale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def __edges(self, image):
        edges = cv2.Canny(image, self.threshold1, self.threshold2)
        return edges

    def __normalize(self, image):
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized

    def __histogram(self, image):
        hist = cv2.calcHist([image], [0], None, [self.bins], [0, 256])
        return hist

    def __resize(self, image):
        resized = cv2.resize(image, self.image_size)
        return resized
