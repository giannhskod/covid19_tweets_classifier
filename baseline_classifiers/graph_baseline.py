.
import csv
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the retweet network as a directed graph
G = nx.read_weighted_edgelist("data/retweet_weighted.edgelist", create_using=nx.DiGraph(), nodetype=int)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Store the ID of the user that posted each message, and initialize for each user a 15-dimensional vector
# that will store the number of messages of each class posted by the user
posted_by = dict()
posts_per_class = dict()
with open('data/posts.tsv', 'r') as f:
    for line in f:
        t = line.split('\t')
        posted_by[int(t[0])] = int(t[1])
        posts_per_class[int(t[1])] = np.zeros(15)

# Read training data. Given a message posted by user A that belongs to class B, increase the number of posts
# of class B posted by user A by 1 
train_index = list()
y_train = list()    
with open('data/train.csv', 'r') as f:
    for line in f:
        t = line.split(',')
        train_index.append(int(t[0]))
        y_train.append(int(t[1]))
        posts_per_class[posted_by[int(t[0])]][int(t[1][:-1])] += 1

# Read test data
test_index = list()  
with open('data/test.csv', 'r') as f:
    for line in f:
        t = line.split(',')
        test_index.append(int(t[0]))

# Create the training matrix. Each row corresponds to a message.
# Use the following 15-dimensional vector of the user that posted the message and concatenate to that vector the
# following two features:
# (1) in-degree of user
# (2) out-degree of user
X_train = np.zeros((len(train_index), 17))
for i,idx in enumerate(train_index):
    for successor in G.successors(posted_by[idx]):
        if successor in posts_per_class:
            X_train[i,:15] += posts_per_class[successor]
    
    for predecessor in G.predecessors(posted_by[idx]):
        if predecessor in posts_per_class:
            X_train[i,:15] += posts_per_class[predecessor]

    if np.sum(X_train[i,:15]) > 0:
        X_train[i,:15] /= np.sum(X_train[i,:15])
    
    X_train[i,15] = G.in_degree(posted_by[idx])
    X_train[i,16] = G.out_degree(posted_by[idx])

# Create the test matrix. Each row corresponds to a message.
# Use the following 15-dimensional vector of the user that posted the message and concatenate to that vector the
# following two features:
# (1) in-degree of user
# (2) out-degree of user
X_test = np.zeros((len(test_index), 17))
for i,idx in enumerate(test_index):
    for successor in G.successors(posted_by[idx]):
        if successor in posts_per_class:
            X_test[i,:15] += posts_per_class[successor]
    
    for predecessor in G.predecessors(posted_by[idx]):
        if predecessor in posts_per_class:
            X_test[i,:15] += posts_per_class[predecessor]

    if np.sum(X_test[i,:15]) > 0:
        X_test[i,:15] /= np.sum(X_test[i,:15])

    X_test[i,15] = G.in_degree(posted_by[idx])
    X_test[i,16] = G.out_degree(posted_by[idx])

# Use logistic regression to classify the messages of the test set
clf = LogisticRegression(solver='liblinear', multi_class='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

# Write predictions to a file
with open('graph_baseline_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = ['id']
    for i in range(15):
        lst.append('class_'+str(i))
    writer.writerow(lst)
    for i,idx in enumerate(test_index):
        lst = y_pred[i,:].tolist()
        lst.insert(0, idx)
        writer.writerow(lst)