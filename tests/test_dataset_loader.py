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
    assert len(result) == 1

    filename, img = result[0]
    assert filename == "car_test.jpg"
    assert img is not None

def test_read_images_ignore_non_image_files(tmp_path):
    loader = DatasetLoader()

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image_path = tmp_path / "car_test.jpg"
    cv2.imwrite(str(image_path), image)

    text_file = tmp_path / "notes.txt"
    text_file.write_text("This is not an image")

    result = loader.read_images(str(tmp_path))

    assert len(result) == 1
    assert result[0][0] == "car_test.jpg"




    