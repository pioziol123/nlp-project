from xgboost import XGBClassifier


class XGBClassifierWrapper:

    def __init__(self):
        self.model = XGBClassifier()

    def fit(self, data, labels):
        self.model.fit(data.tocsc(), labels)

    def predict(self, data):
        return self.model.predict(data.tocsc())
