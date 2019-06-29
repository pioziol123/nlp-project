from collections import Counter
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

class RegexClassifier:
    key_words = {}
    stop_words = None

    def __init__(self):
        stop_words = set(stopwords.words('english'))
        stop_words.add('organization')
        stop_words.add('from')
        stop_words.add('subject')
        stop_words.add('reply-to')
        stop_words.add('nntp-posting-host')

        self.stop_words = stop_words
        self.lemmatizer = WordNetLemmatizer()

    def train(self, data, labels):
        for idx, text in enumerate(data):
            label = labels[idx]
            words = self.__prepare_text(text)
            for counted_word, counter in Counter(words).most_common(20):
                label_counts = self.key_words.get(label, {})
                label_counts[counted_word] = \
                    label_counts.get(counted_word) + counter \
                    if not label_counts.get(counted_word, None) is None else counter
                self.key_words[label] = label_counts
        self.__select_most_used_in_category()

    def __select_most_used_in_category(self):
        for label in self.key_words.keys():
            word_lists = self.key_words[label]
            self.key_words[label] = [word for word in self.__sort_dictionary(word_lists)[-20:]]

    def __prepare_text(self, text):
        words = [word.lower() for word in word_tokenize(text) if len(word) > 3]
        return [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]

    def predict(self, data):
        predictions = []
        for text in data:
            text = ' '.join(self.__prepare_text(text))
            key_words_per_label = {}
            for label in self.key_words.keys():
                key_words_per_label[label] = self.__count_key_words(label, text)
            predictions.append(self.__sort_dictionary(key_words_per_label)[-1:][0])
        return predictions

    def __count_key_words(self, label, text):
        return sum([len(re.findall(word, text)) for word in self.key_words[label]])

    @staticmethod
    def __sort_dictionary(dictionary):
        return sorted(dictionary, key=dictionary.__getitem__)
