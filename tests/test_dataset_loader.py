import os
import cv2
import numpy as np

from src.model.DatasetLoader import DatasetLoader

def test_read_images(tmp_path):
    loader = DatasetLoader()

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image_path = tmp_path / "car_test.jpg"

    cv2.imwrite(str(image_path), image)
    result = loader.read_images(str(tmp_path))
    assert isinstance(result, list)

    