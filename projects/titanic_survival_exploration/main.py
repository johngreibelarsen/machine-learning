# Import libraries necessary for this project
import numpy as np
import pandas as pd


def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred):         
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    else:
        return "Number of predictions does not match number of outcomes!"

    
def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """
    predictions = []
    for _ in data.iterrows():
        predictions.append(0)
    return pd.Series(predictions)


def predictions_1(data):
    """ Model with one feature: sex - a passenger survived if female. """
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            predictions.append(0)            
    return pd.Series(predictions)


def predictions_2(data):
    """ Model with two features: sex - a passenger survived if female.
            age - a passenger survived if they are male and younger than 10. """    
    predictions = []
    for _, passenger in data.iterrows():        
        if passenger['Sex'] == 'female':
            predictions.append(1)
        elif (passenger['Sex'] == 'male') and (passenger['Age'] < 10.0):
            predictions.append(1)
        else:
            predictions.append(0)
    return pd.Series(predictions)


def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex'] == 'female':
            if (passenger['Age'] > 30.0) and (passenger['Pclass'] == 3):
                predictions.append(0)
            else:
                predictions.append(1)
        else: # male block
            if passenger['Age'] < 10.0: 
                predictions.append(1)
            elif ((passenger['Pclass'] in {1,2}) and (passenger['Age'] < 16.0)):
                predictions.append(1)
            else:
                predictions.append(0)
    return pd.Series(predictions)


def main():

    """ Stats:
            #891 in total
            #342 survived
            #577 males
            #314 females
            #32 males < 10 years
    """    

    # Load the dataset
    in_file = 'titanic_data.csv'
    full_data = pd.read_csv(in_file)
    
    # Store the 'Survived' feature in a new variable and remove it from the dataset
    outcomes = full_data['Survived']
    data = full_data.drop('Survived', axis = 1)

    """ Out of the first five passengers, if we predict that all of them survived, 
    what would you expect the accuracy of our predictions to be? Answer: 60% """
    predictions = pd.Series(np.ones(5, dtype = int))
    print accuracy_score(outcomes[:5], predictions)

    """ Always predicts a passenger did not survive """
    predictions = predictions_0(data)
    print accuracy_score(outcomes, predictions) # 1 - (float(outcomes.sum())/outcomes.size)

    """ Always predicts a male as did not survive and a female as survived"""
    predictions = predictions_1(data)
    print accuracy_score(outcomes, predictions)
    
    """ Always predicts a female as survive and a young male under 10 as survived, the rest not """
    predictions = predictions_2(data)
    print accuracy_score(outcomes, predictions)
    
    """ Custom made """
    predictions = predictions_3(data)
    print accuracy_score(outcomes, predictions)
    
    
if __name__ == "__main__": main()