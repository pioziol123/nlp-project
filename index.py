from sklearn.model_selection import train_test_split
from m_l_fitter_factory import MLFitterFactory
from sklearn.datasets import fetch_20newsgroups
from model_evaluator import ModelEvaluator
from regex_classifier import RegexClassifier

fitter = MLFitterFactory().create({'model': 'sgd'})
train_data = fetch_20newsgroups(subset='all', shuffle=True)
x_train, x_test, y_train, y_test = train_test_split(train_data.data, train_data.target, test_size=0.25, random_state=42)

print('\n\n---SGD---\n')
fitter.train(x_train, y_train)
predictions = fitter.predict(x_test)
print(ModelEvaluator.accuracy(predictions, y_test))
print("\n")

print('\n\n---Bayesian---\n')
fitter.model_preceptor.use_bayesian_model()
fitter.train(x_train, y_train)
predictions = fitter.predict(x_test)
print(ModelEvaluator.accuracy(predictions, y_test))

print('\n\n---XGBClassifier---\n')
fitter.model_preceptor.use_xgb_model()
fitter.train(x_train, y_train)
predictions = fitter.predict(x_test)
print(ModelEvaluator.accuracy(predictions, y_test))

print('\n\n---RegexClassifier---\n')
classifier = RegexClassifier()
classifier.train(x_train, y_train)
predictions = classifier.predict(x_test)
print(ModelEvaluator().accuracy(predictions, y_test))
