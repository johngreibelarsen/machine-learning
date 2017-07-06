# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree  import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.metrics import silhouette_score
from IPython.display import display # Allows the use of display() for DataFrames

# Load the wholesale customers dataset
def loadData():
    try:
        data = pd.read_csv("customers.csv")
        data.drop(['Region', 'Channel'], axis = 1, inplace = True)
        print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
        return data
    except:
        print "Dataset could not be loaded. Is the dataset missing?"
        return None


def main():
    print('Loading data...')
    data = loadData()
    display(data.describe())
    
    # TODO: Select three indices of your choice you wish to sample from the dataset
    indices = [8, 100, 300]

    # Create a DataFrame of the chosen samples
    samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
    print "Chosen samples of wholesale customers dataset:"
    display(samples)

    # TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    milk_column = data['Milk']
    new_data = data.drop("Milk", axis = 1)        
    
    
    # TODO: Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(new_data.values, milk_column.values, test_size = 0.25, random_state = 1)        
    
    # TODO: Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state=1)
    regressor.fit(X_train, y_train)
    predictions_test = regressor.predict(X_test)
    # TODO: Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)
    score_accuracy = accuracy_score(y_test, predictions_test.astype(int))
    score_r2 = r2_score(y_test, predictions_test.astype(int), sample_weight=None)
    print(score)
    print(score_accuracy)
    print(score_r2)

    # TODO: Scale the data using the natural logarithm
    log_data = data.apply(lambda x: np.log(x + 1))
    
    # TODO: Scale the sample data using the natural logarithm
    log_samples = samples.apply(lambda x: np.log(x + 1))

    # For each feature find the data points with extreme high or low values
    for feature in log_data.keys():

        # TODO: Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(log_data[feature], 25)
        # TODO: Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(log_data[feature], 75)
        
        # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = 1.5 * (Q3 - Q1)
        print("Step: {}".format(step))
        
        # Display the outliers
        print "Data points considered outliers for the feature '{}':".format(feature)
        display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
        
    # OPTIONAL: Select the indices for data points you wish to remove
    outliers  = []
    
    # Remove the outliers, if any were specified
    good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

    # TODO: Apply PCA by fitting the good data with the same number of dimensions as features
    pca = PCA().fit(good_data)
    print(pca.explained_variance_ratio_) 
    
    # TODO: Transform log_samples using the PCA fit above
    pca_samples = pca.transform(log_samples)

    # TODO: Apply PCA by fitting the good data with only two dimensions
    pca = PCA(n_components = 2).fit(good_data)
    
    # TODO: Transform the good data using the PCA fit above
    reduced_data = pca.transform(good_data)
    
    # TODO: Transform log_samples using the PCA fit above
    pca_samples = pca.transform(log_samples)
    
    # Create a DataFrame for the reduced data
    reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
    print("reduced_data")
    print(reduced_data.get_values()[181])
    print(reduced_data.get_values()[183])
    print(reduced_data.get_values()[338])
    
    
    # TODO: Apply your clustering algorithm of choice to the reduced data 
    #clusterer = KMeans(n_clusters=3, random_state = 1).fit(reduced_data)
    clusterer = GMM(n_components = 3, random_state = 1).fit(reduced_data)
    
    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data) 
    
    # TODO: Find the cluster centers
    #centers = clusterer.cluster_centers_
    centers = clusterer.means_
    
    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples) 
    
    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    score = silhouette_score(reduced_data, preds)
    print(score)

    # TODO: Inverse transform the centers
    log_centers = pca.inverse_transform(centers)
    
    # TODO: Exponentiate the centers
    true_centers = np.exp(log_centers)
    
    # Display the true centers
    segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
    true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
    true_centers.index = segments
    display(true_centers)


if __name__ == "__main__": main()