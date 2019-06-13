class MLFitter:
    vectorizer = None
    data_manager = None
    model_preceptor = None
    data = None
    predictions = None

    def __init__(self, *args, **kwargs):
        self.vectorizer = self.__get_param(kwargs, 'vectorizer')
        self.data_manager = self.__get_param(kwargs, 'data_manager')
        self.model_preceptor = self.__get_param(kwargs, 'model_preceptor')

    @staticmethod
    def __get_param(kwargs, name):
        param = kwargs.get(name, False)
        if not param:
            raise Exception('No %s provided'.format(name))
        return param

    def train(self, data, labels):
        self.data = data
        self.__preprocess_data().__vectorize().__fit_model(labels)

    def predict(self, data):
        self.data = data
        self.__preprocess_data().__vectorize_predictions().__predict()
        return self.predictions

    def __vectorize(self):
        if self.data is None:
            raise Exception('No data for vectorization')
        self.data = self.vectorizer.fit_transform(self.data)
        return self

    def __vectorize_predictions(self):
        if self.data is None:
            raise Exception('No data for vectorization')
        self.data = self.vectorizer.transform(self.data)
        return self

    def __preprocess_data(self):
        if self.data is None:
            raise Exception('No data for preprocessing')
        self.data = self.data_manager.process_data(self.data)
        return self

    def __fit_model(self, labels):
        if self.data is None:
            raise Exception('No data for fitting')
        self.data = self.model_preceptor.fit(self.data, labels)
        return self

    def __predict(self):
        if self.data is None:
            raise Exception('No data for prediction')
        self.predictions = self.model_preceptor.predict(self.data)
