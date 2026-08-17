from abc import ABC, abstractmethod
import numpy as np
import cv2;

class FeatureExtractorStrategy(ABC):

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        pass




    



