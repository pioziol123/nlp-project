from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    remove_stop_words = True
    vectorizer = None
    vectorizers = {
        'tfidf': lambda: TfidfVectorizer()
    }
    is_fitted = False

    def __init__(self, *args, **kwargs):
        vectorizer = self.vectorizers.get(kwargs.get('vectorizer', 'tfidf'), False)
        if not vectorizer:
            raise Exception('No vectorizer found. Existing vectorizers: {tfidf}')
        self.vectorizer = vectorizer()
        self.remove_stop_words = False if not kwargs.get('remove_stop_words', True) else True

    def fit_transform(self, data):
        if self.remove_stop_words:
            self.vectorizer.stop_words = 'english'
        self.is_fitted = True
        return self.vectorizer.fit_transform(data)

    def transform(self, data):
        if not self.is_fitted:
            raise Exception('Vectorizer is not fitted!')
        return self.vectorizer.transform(data)
