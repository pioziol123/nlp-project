class MLFitter:
    vectorizer = None
    data_manager = None
    model_preceptor = None
    data_to_train = None
    data_to_fit = None

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
        if self.data_to_fit is None:
            data = self.__preprocess_data(data)
            data = self.__vectorize(data)
            self.data_to_fit = data
        self.__fit_model(self.data_to_fit, labels)

    def predict(self, data):
        if self.data_to_train is None:
            data = self.__preprocess_data(data)
            data = self.__vectorize_predictions(data)
            self.data_to_train = data
        return self.__predict(self.data_to_train)

    def __vectorize(self, data):
        return self.vectorizer.fit_transform(data)

    def __vectorize_predictions(self, data):
        return self.vectorizer.transform(data)

    def __preprocess_data(self, data):
        return self.data_manager.process_data(data)

    def __fit_model(self, data, labels):
        self.model_preceptor.fit(data, labels)

    def __predict(self, data):
        return self.model_preceptor.predict(data)
