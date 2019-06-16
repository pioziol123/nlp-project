from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from x_g_b_classifier_wrapper import XGBClassifierWrapper


class ModelPreceptor:
    model = None
    fitted = False

    def use_bayesian_model(self):
        self.model = MultinomialNB()

    def use_linear_model(self):
        self.model = LinearRegression()

    def use_sgd_model(self):
        self.model = SGDClassifier()

    def use_xgb_model(self):
        self.model = XGBClassifierWrapper()

    def fit(self, data, labels):
        if self.model is None:
            raise Exception('No model has been chosen')
        self.model.fit(data, labels)
        self.fitted = True

    def predict(self, data):
        if self.model is None:
            raise Exception('No model has been chosen')
        if not self.fitted:
            raise Exception('You have to fit model first')
        return self.model.predict(data)
