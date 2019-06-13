from m_l_fitter import MLFitter
from data_manager import DataManager
from model_preceptor import ModelPreceptor
from vectorizer import Vectorizer


class MLFitterFactory:
    models = {
        'bayesian': lambda p: p.use_bayesian_model(),
        'linear': lambda p: p.use_linear_model(),
        'sgd': lambda p: p.use_sgd_model()
    }

    def create(self, options={}):
        model_preceptor = ModelPreceptor()
        self.choose_model(model_preceptor, options.get('model', 'bayesian'))
        return MLFitter(vectorizer=Vectorizer(), data_manager=DataManager(), model_preceptor=model_preceptor)

    def choose_model(self, model_perceptor, name):
        model = self.models.get(name, False)
        if not model:
            raise Exception("No model found. Available models: {%s}" % self.models.keys())
        model(model_perceptor)
