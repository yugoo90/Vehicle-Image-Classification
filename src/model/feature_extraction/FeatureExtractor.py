

class FeatureExtractor:
    def __init__(self, strategy):
        self.strategy = strategy

    def setStrategy(self, strategy):
        self.strategy = strategy

    def extract(self, image):
        return self.strategy.extract(image)

    def getStrategy(self):
        return self.strategy
