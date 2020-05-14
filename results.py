from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve,average_precision_score
from sklearn import metrics 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def benchmark(model, X_train, y_train, X_test, y_test):
    
    #fit the NN model to the data
    model.fit(
    X_train,                      
    y_train,
    epochs=20,
    batch_size=64,
    verbose=0,
    validation_split=0.2
    )
    
    #get predictions, based on which all metrics will be calculated
    predict = model.predict_classes(np.array(X_test))
    f1_score = metrics.f1_score(y_test, predict, average='weighted')
    acc = metrics.accuracy_score(y_test, predict)
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    result = {'f1_score' : f1_score, 'accuracy' : acc, 'train_size' : len(y_train),
              'test size' : len(y_test), 'recall': recall, 'precision': precision,
              'predictions': predict }
    
    return result

def get_results(model, X_train, label_train, X_test, label_test):
    #get copies of the initial data so as to avoid any change
    train_x_s_s, train_y_s_s = X_train[:], label_train[:]
    test_x_s_s, test_y_s_s = X_test[:], label_test[:]
    
    #initialize a dictionary called 'results', inside which all metrics will be stored
    results = {}
    results['train_size'] = []
    results['accuracy_on_test'] = []
    results['accuracy_on_train'] = []
    results['F1_on_train'] = []
    results['F1_on_test'] = []
    results['recall_on_train'] = []
    results['recall_on_test'] = []
    results['precision_on_train'] = []
    results['precision_on_test'] = []
    
    #gradually increase the initial chunk of data
    for i in range(1, 11):
        if(i==10):
            train_x_part = train_x_s_s
            train_y_part = train_y_s_s
        else:
            to = int(i*(train_x_s_s.shape[0]/10))         
            train_x_part = train_x_s_s[0: to, :]
            train_y_part = train_y_s_s[0: to]
            
        #call benchmark function for both test and training and feed it with data up to this point
        results['train_size'].append(train_x_part.shape[0])
        result = benchmark(model, train_x_part, train_y_part, 
                           test_x_s_s, test_y_s_s)
        results['accuracy_on_test'].append(result['accuracy'])
        results['F1_on_test'].append(result['f1_score'])
        results['recall_on_test'].append(result['recall'])
        results['precision_on_test'].append(result['precision'])
        
        result = benchmark(model, train_x_part, train_y_part, 
                           train_x_part, train_y_part)
        results['accuracy_on_train'].append(result['accuracy'])
        results['F1_on_train'].append(result['f1_score'])
        results['recall_on_train'].append(result['recall'])
        results['precision_on_train'].append(result['precision'])
        
    return results

def learning_curves(results,graph_type, y_lower=0.94):
    #adjust the output graph depending on whether it is an accuracy, f1-score,precision or recall
    #learning curves graph
    if graph_type == 'accuracy':
        label1 = 'Accuracy on Train'
        label2 = 'Accuracy on Test'
        result1 = results['accuracy_on_train']
        result2 = results['accuracy_on_test'] 
        label_y = 'Accuracy'
    elif  graph_type == 'F1-score':
        label1='F1-score on Train'
        label2='F1-score on Test'
        result1 = results['F1_on_train']
        result2 = results['F1_on_test'] 
        label_y = 'F1-score'
    elif  graph_type == 'precision':
        label1='precision on Train'
        label2='precision on Test'
        result1 = results['precision_on_train']
        result2 = results['precision_on_test'] 
        label_y = 'precision'
    else:
        label1='recall on Train'
        label2='recall on Test'
        result1 = results['recall_on_train']
        result2 = results['recall_on_test'] 
        label_y = 'recall'
        
    #create graph
    pylab.rcParams['figure.figsize'] = (20, 6)
    fontP = FontProperties()
    fontP.set_size('small')
    fig = plt.figure()
    fig.suptitle('Learning Curves', fontsize=20)
    ax = fig.add_subplot(111)
    #the lowest value of the y axis can be set by the user
    ax.axis([350, 4250, y_lower, 1.01])
    line_up, = ax.plot( results['train_size'], result1, 'o-',label=label1)
    line_down, = ax.plot( results['train_size'] ,result2, 'o-',label=label2)

    plt.xlabel('N. of training instances', fontsize=18)
    plt.ylabel(label_y, fontsize=16)
    plt.legend([line_up, line_down], [label1, label2], prop = fontP)
    plt.grid(True)


def conf_matrix(label_test, predictions):
    con_matrix = confusion_matrix(label_test, predictions)
    print(con_matrix)
    plt.matshow(con_matrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def receiver_curv(label_test, predictions):
    false_positive_rate, recall, thresholds = roc_curve(label_test, predictions)
    roc_auc = auc(false_positive_rate, recall)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.show()


def prec_rec_curve(model, X_test, label_test):
    y_score = model.predict_proba(np.array(X_test))
    average_precision = average_precision_score(label_test, y_score)
    precision, recall, _ = precision_recall_curve(label_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))