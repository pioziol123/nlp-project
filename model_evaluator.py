from sklearn.metrics import accuracy_score


class ModelEvaluator:
    @staticmethod
    def accuracy(predictions, labels):
        print(predictions)
        print(labels)
        return accuracy_score(labels, predictions) * 100
