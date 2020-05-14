# k may be either an integer greater than zero
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

def select_features_pca(train_X, test_X, k):
    selector = PCA(n_components=k)
    selector.fit(train_X)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)
    return train_X, test_X


def get_data_set():
	# Read training data
	train_index = list()
	y_train = list()   
	y_pred = list()
	y_train_as_is = list()
	with open('data/train.csv', 'r') as f:
	    for line in f:
	        t = line.split(',')
	        train_index.append(int(t[0]))
	        y_train.append([int(t[1]),])
	        y_train_as_is.append(int(t[1]))
	# Read test data
	test_index = list()  
	with open('data/test.csv', 'r') as f:
	    for line in f:
	        t = line.split(',')
	        test_index.append(int(t[0]))
	        # y_pred.append([int(t[1]),])

	# Load the textual content of the messages into the dictionary "posts"
	posts = dict()
	with open('data/posts.tsv', 'r') as f:
	    for line in f:
	        t = line.split('\t')
	        posts[int(t[0])] = t[2][:-1]

	# Create 2 lists: one containing the messages of the training set and the other containing the messages of the
	# test set
	train_posts = [posts[idx] for idx in train_index]
	test_posts = [posts[idx] for idx in test_index]

	mlb = MultiLabelBinarizer()
	y_train = mlb.fit_transform(y_train)

	return train_posts, test_posts, y_train, y_pred, y_train_as_is, test_index


def export_results(test_index,y_pred):
	# Write predictions to a file
	with open('text_baseline_submission.csv', 'w') as csvfile:
	    writer = csv.writer(csvfile, delimiter=',')
	    lst = ['id']
	    for i in range(15):
	        lst.append('class_'+str(i))
	    writer.writerow(lst)
	    for i,idx in enumerate(test_index):
	        lst = y_pred[i,:].tolist()
	        lst.insert(0, idx)
	        writer.writerow(lst)