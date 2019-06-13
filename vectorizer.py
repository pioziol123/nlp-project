from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


class Vectorizer:
    remove_stop_words = True
    vectorizer = None
    vectorizers = {
        'tfidf': lambda: TfidfTransformer()
    }
    vectorized_data = None
    count_vectorizer = CountVectorizer()
    is_fitted = False

    def __init__(self, *args, **kwargs):
        vectorizer = self.vectorizers.get(kwargs.get('vectorizer', 'tfidf'), False)
        if not vectorizer:
            raise Exception('No vectorizer found. Existing vectorizers: {tfidf}')
        self.vectorizer = vectorizer()
        self.remove_stop_words = False if not kwargs.get('remove_stop_words', True) else True

    def fit_transform(self, data):
        if self.is_fitted:
            return self.transform(data)
        if self.remove_stop_words:
            self.count_vectorizer.stop_words = 'english'
        self.vectorized_data = self.vectorizer.fit_transform(self.count_vectorizer.fit_transform(data))
        self.is_fitted = True
        return self.vectorized_data

    def transform(self, data):
        self.vectorized_data = self.vectorizer.transform(self.count_vectorizer.transform(data))
        return self.vectorized_data

    def get_vectorized_data(self):
        return self.vectorized_data
