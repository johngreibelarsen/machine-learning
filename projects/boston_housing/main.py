import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
# version 0.17 below for train_test_split
from sklearn.cross_validation import train_test_split
# version 0.18 below for train_test_split
#from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from numpy import ndarray

def predictTrials(X, y, fitter, data):
    """ Performs trials of fitting and predicting data. """

    print("\nViewing robustness of predictions:")
    
    prices = []
    for k in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = k)
        reg = fitter(X_train, y_train)
        
        # Make a prediction on the first element only
        pred = reg.predict([data[0]])[0]
        prices.append(pred)
        
        print "Trial {}: ${:,.2f}".format(k+1, pred)

    print "Range in prices: ${:,.2f}".format(max(prices) - min(prices))
    ndPrices = np.array(prices)
    print "Mean price: ${:,.2f}".format(ndPrices.median())
    print "Standard Variation: ${:,.2f}".format(ndPrices.std())
    ndPrices.min()
    
        
def performance_metric(X, y):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """    
    score = r2_score(X, y)
    return score


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """    

    #  scikit-learn versions 0.17 (below)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    #  scikit-learn versions 0.18 (below)
    # cv_sets = ShuffleSplit(X.shape[0], n_splits = 10, test_size = 0.20, random_state = 0)

    regressor = DecisionTreeRegressor(random_state = 1)

    # Dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': range(1, 11)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, y)

    return grid.best_estimator_


def main():
    # Load the Boston housing dataset
    data = pd.read_csv('housing.csv')
    prices = data['MEDV']
    features = data.drop('MEDV', axis = 1)
        
    print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)    
    print "Min price: ${:,.2f}".format(prices.min())
    print "Max price: ${:,.2f}".format(prices.max())
    print "Mean price: ${:,.2f}".format(prices.mean())
    print "Median price: ${:,.2f}".format(prices.median())
    print "Standard variation: ${:,.2f}".format(prices.std())
    
    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 1)
        
    reg = fit_model(X_train, y_train)
    
    # Find the value for 'max_depth'
    print "\nParameter 'max_depth' is {} for the optimal model.\n".format(reg.get_params()['max_depth'])
    
    # Matrix for some selected clients that want a valuation
    client_data = [[5, 17, 15], # Client 1
                   [4, 32, 22], # Client 2
                   [8, 3, 12]]  # Client 3
    
    # Show predictions
    for i, price in enumerate(reg.predict(client_data)):
        print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)
    
    predictTrials(features, prices, fit_model, client_data)
    
    
if __name__ == "__main__": main()