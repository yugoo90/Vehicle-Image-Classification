from abc import ABC, abstractmethod
import numpy as np

class VehicleClassifierStrategy(ABC):

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    

