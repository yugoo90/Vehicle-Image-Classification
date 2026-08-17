import string


class LabelEncoder:
    def __init__(self, images):
        self.label_map = {
            "car": 0, 
            "suv": 1,
            "bus": 2,
        }


    # map the image dataset to a label class
    def label(self):
        label = []
        for filename, img in self.images:
            label.append(self.get_label(filename, self.label_map))
        return label


    # get the label of the image
    def get_label(self, filename, label_map):
        name = filename.lower()
        for key in label_map:
            if key in name:
                return self.label_map[key]    
        return None