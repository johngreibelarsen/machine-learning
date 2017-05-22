import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV 

def load_data():
    # Load the Census dataset
    data = pd.read_csv("census.csv")    
    return data

def preprocess_data(data):
    labels = data.income.map({'<=50K':0, '>50K':1})    
    features = data.drop('income', axis = 1)
    
    # Log-transform the skewed features
    skewed = ['capital-gain', 'capital-loss']
    features[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
    
    # Normalize each numerical feature
    scaler = MinMaxScaler()
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    features[numerical] = scaler.fit_transform(data[numerical])    
    
    # One-hot encode the 'features_raw' data
    features = pd.get_dummies(features)
    encoded = list(features.columns)
    print "{} total features after one-hot encoding.".format(len(encoded))

    return (features, labels)

def initial_analyze(features, labels):
    n_records = features.shape[0]
    n_greater_50k = labels.sum()
    n_at_most_50k = n_records - n_greater_50k
    greater_percent = 100 * float(n_greater_50k) / n_records
    
    # Print the results
    print "Total number of records: {}".format(n_records)
    print "Individuals making more than $50,000: {}".format(n_greater_50k)
    print "Individuals making at most $50,000: {}".format(n_at_most_50k)
    print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)    


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, 0.5)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)
       
    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
    print "Results are: {}.".format(results)
        
    # Return the results
    return results


def main():
    print("*** Main Starting***")
    
    data = load_data()
    modified_data = preprocess_data(data)
    features = modified_data[0]
    income = modified_data[1]
    initial_analyze(features, income)
    
    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)
    # Show the results of the split
    print "Training set has {} samples.".format(X_train.shape[0])
    print "Testing set has {} samples.".format(X_test.shape[0])

    print("/n*** Running Models ***")
    # TODO: Initialize the three models
    clf_A = RandomForestClassifier()
    clf_B = DecisionTreeClassifier()
    clf_C = SVC(kernel="linear")
    clf_D =  KNeighborsClassifier()
    
    
    # TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
    samples_1 = int(round(X_train.shape[0]*0.01))
    samples_10 = int(round(X_train.shape[0]*0.1))
    samples_100 = X_train.shape[0]
    print(type(samples_1))
    """    
    # Collect results on the learners
    results = {}
    for clf in [clf_A]: #clf_A, clf_B, clf_C, clf_D
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]): # samples_1, samples_10, samples_100
            results[clf_name][i] = \
            train_predict(clf, samples, X_train, y_train, X_test, y_test)


    # Fine tune the Rabndom Forest Trees model

    # TODO: Initialize the classifier
    clf = RandomForestClassifier(max_features=0.6)
    
    # TODO: Create the parameters list you wish to tune
    parameters = {'n_estimators':(50, 60, 70), 'max_depth':(15, 16, 17)}
    
    # TODO: Make an fbeta_score scoring object
    scorer = make_scorer(fbeta_score, beta=0.5)
    
    # TODO: Perform grid search on the classifier using 'scorer' as the scoring method
    grid_obj = GridSearchCV(clf, param_grid = parameters, scoring = scorer, verbose = 10)
    
    # TODO: Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_obj.fit(X_train, y_train)
    
    # Get the estimator
    best_clf = grid_fit.best_estimator_
    print("Best params:")
    print(grid_fit.best_params_)
    """    
    # Make predictions using the unoptimized and model
    predictions = (clf_A.fit(X_train, y_train)).predict(X_test)
    #best_predictions = best_clf.predict(X_test)

    # Report the before-and-afterscores
    print "Unoptimized model\n------"
    print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
    print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
    print "\nOptimized Model\n------"
    #print("Best params: {}").format(grid_fit.best_params_)
    #print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
    #print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))

    print("*** Main finished ***")

    importances = clf_A.feature_importances_
    print( importances )
    print( np.argsort(importances) )
    print( (np.argsort(importances)[::-1]) )
    print( (np.argsort(importances)[::-1])[:5] )
    
    print("\n")
    x = np.array([3, 5, 2, 1, 4, 12])
    print(np.argsort(x))

if __name__ == "__main__":
    main()
