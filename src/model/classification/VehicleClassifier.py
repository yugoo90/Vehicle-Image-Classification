from model.feature_extraction.FeatureExtractor import FeatureExtractor

class VehicleClassifier:
    def __init__(self, classifier):
        self.classifier = classifier

    def setStrategy(self, strategy):
        self.classifier = strategy

    def classify(self, featureExtractor: FeatureExtractor):
        return self.classifier.classify(featureExtractor.extract())

    def getStrategy(self):
        return self.classifier
