from model.DatasetLoader import DatasetLoader
from model.feature_extraction.EdgeHistogram import EdgeHistogram
from model.feature_extraction.FeatureExtractor import FeatureExtractor



dataset_loader = DatasetLoader()

images = dataset_loader.read_images("../data")

feature_extractor = FeatureExtractor(EdgeHistogram())


for filename, image in images:
    feature = feature_extractor.extract(image)
    print("File: ", filename)
    print("Feature: ", feature)
    print("Feature count: ", len(feature))
    print("Feature sum: ", feature.sum())
    print()





