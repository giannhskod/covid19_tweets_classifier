import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Read training data
train_index = list()
y_train = list()    
with open('data/train.csv', 'r') as f:
    for line in f:
        t = line.split(',')
        train_index.append(int(t[0]))
        y_train.append(int(t[1]))

# Read test data
test_index = list()  
with open('data/test.csv', 'r') as f:
    for line in f:
        t = line.split(',')
        test_index.append(int(t[0]))

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

# Create the training matrix. Each row corresponds to a message and each column to a word present in at least 5
# messages of the training set. The value of each entry in a row is equal to the tf-idf weight of that word in the 
# corresponding message 
vectorizer = TfidfVectorizer(min_df=5)
X_train = vectorizer.fit_transform(train_posts)

# Create the test matrix following the same approach as in the case of the training matrix
X_test = vectorizer.transform(test_posts)

print("Train matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)

# Use logistic regression to classify the messages of the test set
clf = LogisticRegression(solver='newton-cg', multi_class='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

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