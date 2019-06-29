import concurrent.futures

from nltk import LancasterStemmer, PorterStemmer, WordNetLemmatizer, SnowballStemmer, word_tokenize


class DataManager:
    lemmatize = True
    stem = False
    stemmer = None
    lemmatizer = None
    remove_short_words = True
    stemmers = {
        'lancaster': lambda: LancasterStemmer(),
        'porter': lambda: PorterStemmer(),
        'snowball': lambda: SnowballStemmer('english')
    }
    lemmatizers = {
        'word_net': lambda: WordNetLemmatizer(),
    }

    parallelize = True

    processed_data = None

    def __init__(self, *args, **kwargs):
        if kwargs.get('lemmatize', None) is not None:
            self.lemmatize = False if not kwargs.get('lemmatize', True) else True
        elif kwargs.get('stem', None) is not None:
            self.stem = False if not kwargs.get('stem', True) else True
        self.choose_lemmatizer(kwargs.get('lemmatizer', 'word_net'))
        self.choose_stemmer(kwargs.get('stemmer', 'porter'))
        self.remove_short_words = False if not kwargs.get('remove_short_words', True) else True
        self.parallelize = False if not kwargs.get('parallelize', True) else True

    def choose_stemmer(self, stemmer_name):
        if not self.stem:
            return
        stemmer = self.stemmers.get(stemmer_name, False)
        if not stemmer:
            self.stemmer = stemmer()
        else:
            raise Exception('No stemmer found. Existing stemmers: {lancaster, porter, snowball}')

    def choose_lemmatizer(self, lemmatizer_name):
        if not self.lemmatize:
            return
        lemmatizer = self.lemmatizers.get(lemmatizer_name, False)
        if not lemmatizer:
            raise Exception('No lemmatizer found. Existing lemmatizers: {word_net}')
        self.lemmatizer = lemmatizer()

    def process_data(self, data):
        self.processed_data = []

        # if self.parallelize:
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        #         futures = [executor.submit(self.__process_data, text) for text in data]
        #         for idx, future in enumerate(concurrent.futures.as_completed(futures)):
        #             self.processed_data.append(future.result())
        # else:
        for text in data:
            self.processed_data.append(self.__process_data(text))
        return self.processed_data

    def __process_data(self, text):
        words = word_tokenize(text)
        if self.remove_short_words:
            words = [word for word in words if len(word) > 3]
        words = [word.lower() for word in words]
        if self.stem:
            words = [self.stemmer.stem(word) for word in words]
        if self.lemmatize:
            words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def get_processed(self):
        return self.processed_data
