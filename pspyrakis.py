import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import models, helpers, results
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve,average_precision_score
from sklearn import metrics 
# Read training data
train_posts, test_posts,label_train, label_test, y_train_as_is, test_index = helpers.get_data_set()

print('Read DONE ')

vectorizer = TfidfVectorizer(stop_words='english',min_df=5)
X_train_transformer = vectorizer.fit_transform(train_posts)
X_train = X_train_transformer.toarray()

X_test_transformer = vectorizer.transform(test_posts)
X_test = X_test_transformer.toarray()
print('tfidf DONE ')

#data normalization 
X_train_init,X_test_init=X_train[:],X_test[:]
X_train = preprocessing.normalize(X_train, norm='l2')
X_test = preprocessing.normalize(X_test, norm='l2')
print('normalization DONE ')

# PCA
k = 0.6
train_x_sub, test_x_sub = helpers.select_features_pca(X_train, X_test, k)
print('PCA DONE')

print('previous train shape : ')
print(X_train_init.shape)
print('new train shape : ')
print(train_x_sub.shape)
print('previous test shape :')
print(X_test_init.shape)
print('new test shape :')
print(test_x_sub.shape)

# model training
mlp_model = models.MLP(train_x_sub)
history = mlp_model.fit(
    train_x_sub,                      
    label_train,
    epochs=100,
    batch_size=64,
    verbose=0,
    validation_split=0.2
)

score_train = mlp_model.evaluate(
    train_x_sub,
    label_train,
    batch_size=64,
    verbose=0
)

# score_test = mlp_model.evaluate(
#     test_x_sub,
#     label_test,
#     batch_size=64,
#     verbose=0
# )

    
print('categorical_accuracy of training data : {0:0.3f} '.format(score_train[0]))
print('Accuracy on training data : {0:0.3f} %'.format(100*score_train[1]))
# print('Binary_crossentropy of test data : {0:0.3f} '.format(score_test[0]))
# print('Accuracy on test data : {0:0.3f} %'.format(100*score_test[1]) )

# mlp_test_predictions = mlp_model.predict_classes(np.array(test_x_sub))
# mlp_precision = metrics.precision_score(label_test, mlp_test_predictions)
# mlp_recall = metrics.recall_score(label_test, mlp_test_predictions)
# mlp_test_f1_score =  metrics.f1_score(label_test, mlp_test_predictions)
# mlp_test_accuracy = metrics.accuracy_score(label_test, mlp_test_predictions)

mlp_train_proba = mlp_model.predict_proba(np.array(train_x_sub))
mlp_train_predictions = mlp_model.predict_classes(np.array(train_x_sub))

mlp_train_f1_score =  metrics.f1_score(y_train_as_is, mlp_train_predictions,average = "micro")
mlp_train_accuracy = metrics.accuracy_score(y_train_as_is, mlp_train_predictions)

# print('Precision on test: {0:0.3f} %'.format(100*mlp_precision) )
# print('Recall on test: {0:0.3f} %'.format(100*mlp_recall))
print('F1 score on test: {0:0.3f} %'.format(100*mlp_train_f1_score))
print('Accuracy score on test: {0:0.3f} %'.format(100*mlp_train_accuracy))
helpers.export_results(test_index, mlp_train_proba)

