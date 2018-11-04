import pandas as pd
import numpy as np
import os
import sys
from timeit import default_timer as timer

from pandas import read_csv
from pandas import DataFrame as df
from pandas import read_csv
from pandas import DataFrame as df

import datetime as dt
import json
from pandas.io.json import json_normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.datasets import fetch_mldata

from pylab import *
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.decomposition import FastICA, PCA,RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn import random_projection

from sklearn.neural_network import MLPClassifier

def plot_confusion_matrix(target_names,cm,title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the '+title)
    fig.colorbar(cax)
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    #plt.xlabel('Predicted')
    plt.ylabel('True')
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    plt.xlabel('Predicted \naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def startTime():
    return timer()

def endTime():
    return timer()

def timeTaken(a,b,str):
    print (str)
    print ('time taken {:0.4f}'.format(b - a))

def checkOptimaldimensionality(s):
    # range of distortions
    eps_range = np.linspace(0.1, 0.99, 10)
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(eps_range)))

    # range of number of samples (observation) to embed
    n_samples_range = np.logspace(1, 4, s)

    plt.figure()
    for eps, color in zip(eps_range, colors):
        min_n_components = johnson_lindenstrauss_min_dim(n_samples_range, eps=eps)
        plt.loglog(n_samples_range, min_n_components, color=color)
    plt.legend(["eps = %0.1f" % eps for eps in eps_range], loc="lower right")
    plt.xlabel("Number of observations to eps-embed")
    plt.ylabel("Minimum number of dimensions")
    plt.title("Johnson-Lindenstrauss bounds:\nn_samples vs n_components")
    plt.legend(["eps = %0.1f" % eps for eps in eps_range], loc="lower right")
    plt.xlabel("Number of observations to eps-embed")
    plt.ylabel("Minimum number of dimensions")
    plt.title("Johnson-Lindenstrauss bounds:\nn_samples vs n_components w.r.t eps")

def kMeans(n, X, index1, index2):
    # 0 and 1 are usually passed for start index and end index
    print ("In kmeans")
    kmeans = KMeans(n_clusters=n)
    a = startTime()
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    b = endTime() 
    timeTaken(a,b,"K-Means")

    centers = kmeans.cluster_centers_
    #plt.scatter(centers[:, index1], centers[:, index1], c='black', s=200, alpha=0.5);
    #plt.show()
     # because first time the call is the data frame [[]], cluster + feature transform 2nd time array
    if isinstance(X,(np.ndarray)):
        plt.scatter(X[:, index1], X[:, index2],c=y_kmeans, s=50, cmap='viridis')
    else:
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1],c=y_kmeans, s=50, cmap='viridis')
    plt.show()
    return kmeans

def gmmEM(n, X, index1, index2):
    # 0 and 1 are usually passed for start index and end index
    gmm = GaussianMixture(n_components=n)
    a = startTime()
    gmm.fit(X)
    y_gmm = gmm.predict(X)
    b = endTime() 
    timeTaken(a,b,"GMM")
    print(gmm.means_)
    print('\n')
    print(gmm.covariances_)
    
    #if (index2 == 2):
     #   plt.scatter(X['channelGrouping'], X['geoNetwork.metro'],c=y_gmm, s=50, cmap='viridis')
    #else:
    # because first time the call is the data frame [[]], cluster + feature transform 2nd time array
    if isinstance(X,(np.ndarray)):
        plt.scatter(X[:, index1], X[:, index2],c=y_gmm, s=50, cmap='viridis')
    else:
        plt.scatter(X.iloc[:, index1], X.iloc[:, index2],c=y_gmm, s=50, cmap='viridis')
    plt.show()
   
def findOptimumCluster(X):      
    n_components = np.arange(1,19)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
          for n in n_components]
    plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');
    plt.show()
 
def load_df(nrows=1000):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    my_path = os.path.abspath(os.path.dirname(__file__))
    path_train = os.path.join(my_path, "data\\GstoreCustomer\\train.csv")
    
    data_train = pd.read_csv(path_train)
   
    df = pd.read_csv(path_train, 
                     converters = {column: json.loads for column in JSON_COLUMNS}, 
                     dtype = {'fullVisitorId': 'str'}, 
                     nrows = nrows)
        
    for column in JSON_COLUMNS:
       column_as_df = json_normalize(df[column])
       column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
       df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
   
    # cleanse
    # to numeric values
    df['totals.transactionRevenue'] = pd.to_numeric(df['totals.transactionRevenue'])
    df['totals.transactionRevenue'] = df['totals.transactionRevenue'].fillna(0)
    df['totals.hits'] = pd.to_numeric(df['totals.hits']).fillna(0)
    df['totals.pageviews'] = pd.to_numeric(df['totals.pageviews']).fillna(0)
    # dates from int to Timestamp
    df['date'] = pd.to_datetime(df.date, format='%Y%m%d')
    # drop useless colums that have only 1 value
    train_const_cols = [col for col in df.columns if len(df[col].unique()) == 1]
    df.drop(train_const_cols, axis = 1, inplace = True)
    # categorical features remove - why?
    cat_feat = list(df.columns.values)
    #cat_feat.remove('totals.transactionRevenue')
    cat_feat.remove("totals.pageviews")
    cat_feat.remove("totals.hits")
    # numerical features
    num_feat = ["totals.hits", "totals.pageviews"]
    print (df.dtypes)
    # normalize features, convert text to string, convert mumeric to float type
    for feat in cat_feat:
        lbl = LabelEncoder()
        lbl.fit(list(df[feat].values.astype('str')))
        df[feat] = lbl.transform(list(df[feat].values.astype('str')))
            
    for feat in num_feat:
        df[feat] = df[feat].astype(float)
    feats= cat_feat + num_feat
  
    return df[feats]

def variancePCA(X):
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('# of components')
    plt.ylabel('cumulative explained variance');
    plt.show()

def varianceRandomizedPCA(X):
    rpca = RandomizedPCA().fit(X)
    plt.plot(np.cumsum(rpca.explained_variance_ratio_))
    plt.xlabel('# of components')
    plt.ylabel('cumulative explained variance');
    plt.show()

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = None)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    
def callPCA(X, n, type):
    # type = 1 for Energy data to avoid 1D plot, 2 for others, type 3 used for image reconstruct plot
    # n = # of component or % of variance
    print('In PCA')
    pca = PCA(n_components = n)
    if n < 1:
        pca = PCA(n)     
    a = startTime()
    pca.fit(X)
    X_pca = pca.transform(X)
    b = endTime() 
    timeTaken(a,b,"PCA transform")
    print("original shape:   ", X.shape)
    print("transformed shape after PCA:", X_pca.shape)
    a = startTime()
    X_recons = pca.inverse_transform(X_pca)
    b = endTime() 
    timeTaken(a,b,"PCA reconstruct")
    print("reconstruct shape after PCA:", X_recons.shape)

    if type == 2: # Gstore data
        myplot(X_pca[:,0:2],np.transpose(pca.components_[0:2, :]))
        plt.show()
        myplot(X_recons[:,0:2],np.transpose(pca.components_[0:2, :]))
        plt.show()
    
    X_std = StandardScaler().fit_transform(X)
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print('Eigenvectors \n%s' %eig_vecs)
    print('\nEigenvalues \n%s' %eig_vals)
    if type == 1: # Energy data
        sns.set(color_codes=True)
        sns.distplot(eig_vals) 
    else:
        #sns.distplot gives type cast error for Gstore, follow the histogram
        plt.hist(eig_vals)
    plt.show()

    if type == 3: #image recostruct 
        plt.figure(figsize=(8,4));
        mnist = fetch_mldata('MNIST original')   
        # Original Image
        plt.subplot(1, 2, 1);
        plt.imshow(mnist.data[8].reshape(28,28),
                      cmap = plt.cm.gray, interpolation='nearest',
                      clim=(0, 255));
        plt.xlabel('784 components', fontsize = 14)
        plt.title('Original Image', fontsize = 20);
        plt.show()

        # 154 principal components
        plt.subplot(1, 2, 2);
        plt.imshow(X_recons[8].reshape(28, 28),
                      cmap = plt.cm.gray, interpolation='nearest',
                      clim=(0, 255));
        plt.xlabel('154 components', fontsize = 14)
        plt.title('95% of Explained Variance', fontsize = 20);
        plt.show()
           
    return X_pca

def callICA(X,n,type): 
    # type = 1 for Energy data to avoid 1D plot, 2 for others
    ica = FastICA(n_components=n)
    a = startTime()
    X_transformed = ica.fit_transform(X)  
    b = endTime() 
    timeTaken(a,b,"ICA transform")
    print("original shape:   ", X.shape)
    print("transformed shape after ICA:", X_transformed.shape)
    a = startTime()
    X_recons = ica.inverse_transform(X_transformed)
    b = endTime() 
    timeTaken(a,b,"ICA transform")
    print("reconstruct shape after ICA:", X_recons.shape)

    if type > 1: # Gstore data
        myplot(X_transformed[:,0:2],np.transpose(ica.components_[0:2, :]))
        plt.show()
        myplot(X_recons[:,0:2],np.transpose(ica.components_[0:2, :]))
        plt.show()
    
        X_std = StandardScaler().fit_transform(X_transformed)
        mean_vec = np.mean(X_std, axis=0)
        cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
        cov_mat = np.cov(X_std.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        print('Eigenvectors \n%s' %eig_vecs)
        print('\nEigenvalues \n%s' %eig_vals)
        if type == 1: # Energy data
            sns.set(color_codes=True)
            sns.distplot(eig_vals) 
        else:
            #sns.distplot gives type cast error for Gstore, follow the histogram
            plt.hist(eig_vals)
        plt.show()
    
    return X_transformed

def callRandomProjection(X,n,type):  
    # type = 1 for Energy data to avoid 1D plot, 2 for others
    transformer = random_projection.SparseRandomProjection(n_components=n)
    a = startTime()
    X_new = transformer.fit_transform(X) 
    b = endTime() 
    timeTaken(a,b,"PCA transform")
    print("original shape:   ", X.shape)
    print("transformed shape after RandomizedProjection:", X_new.shape)
    # can't reconstruct
    # very few components are non-zero
    print("non zero components NZC must be minimal, check by NZC low mean: ",np.mean(transformer.components_ != 0))

    if type == 2: # Gstore data
        myplot(X_new[:,0:2],np.transpose(transformer.components_[0:2, :]))
        plt.show()
    
    return X_new

def callRandomizedPCA(X,n,type):    
    # type = 1 for Energy data to avoid 1D plot, 2 for others
    rpca = RandomizedPCA(n_components=n)
    rpca.fit(X)
    transformed = rpca.transform(X)
    print("original shape:   ", X.shape)
    print("transformed shape after Randomized PCA:", transformed.shape)
    X_recons = rpca.inverse_transform(transformed)
    print("reconstruct shape after Randomized PCA:", X_recons.shape)
    
    if type == 2: # Gstore data
        myplot(transformed[:,0:2],np.transpose(rpca.components_[0:2, :]))
        plt.show()
        myplot(X_recons[:,0:2],np.transpose(rpca.components_[0:2, :]))
        plt.show()

    return transformed

def findNeuralNwOptimalVars(X_train, X_test, y_train, y_test):
    error = []
    # Calculating error for various number of hidden layers
    for a in range(1,100):             
        mlp = MLPClassifier(hidden_layer_sizes=(a),max_iter=1500)
        mlp.fit(X_train,y_train)
        y_pred = mlp.predict(X_test) 
        error.append(np.mean(y_pred != y_test))
    plt.figure(figsize=(12, 6))  
    plt.plot(range(1,100), error, color='red', linestyle='dashed', marker='o',  
    markerfacecolor='blue', markersize=10)
    plt.xlabel('number of hidden layers') 
    plt.title('Error Rate Neural network')  
    plt.ylabel('Mean Error')  
    plt.show()

def callNeuralNwClassifier(X,X1):
    # supervised classification, need original data set X to get tartget variable
    # new data set X1 - dimension reduced, clustered on reduced dimension augmented, etc.
    print ("In neural nw")
    y = X['channelGrouping']    
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, shuffle=True, random_state=0)
    findNeuralNwOptimalVars(X_train, X_test, y_train, y_test)
    hiddenLayerSize = 12
    mlp = MLPClassifier(hidden_layer_sizes=(hiddenLayerSize),max_iter=1500)
    a = startTime()
    mlp.fit(X_train,y_train)
    y_pred = mlp.predict(X_test)
    b = endTime() 
    timeTaken(a,b,"Neural Network")
    cm = confusion_matrix(y_test,y_pred)
    plot_confusion_matrix(np.unique(y_pred),cm,"Neural network MLPClassifiction")
    print("Neural NW MLP confusion matrix {}".format(confusion_matrix(y_test,y_pred)))
    print("Neural NW MLP classification report {}".format(classification_report(y_test,y_pred)))
      
def callEnergyData(): 
    my_path = os.path.abspath(os.path.dirname(__file__))
    path_train = os.path.join(my_path, "data\\industrial_data.csv")    
    df = pd.read_csv(path_train) 
    print ("============= Energy Data ===========================")
    print (df.dtypes)
    print (df.head(2))     
    X2 = df
    clusterOptimum = 5
    findOptimumCluster(X2)
    gmmEM(clusterOptimum, X2, 0, 1)
    kMeans(clusterOptimum, X2, 0, 1)
    
    # run dimension reduction
    s=16382
    checkOptimaldimensionality(s)
    variancePCA(X2)
    nc = 1
    x_pca = callPCA(X2, nc, 1)
    x_ica = callICA(X2,nc,1)
    x_randomProj = callRandomProjection(X2,nc,1)
    varianceRandomizedPCA(X2)
    x_randomizedPCA = callRandomizedPCA(X2,nc,1)

    # re-run clusters not needed on two data sets
    #findOptimumCluster(x_pca)
    #gmmEM(clusterOptimum, x_pca,0,1)
    #findOptimumClusterKmeans(x_pca)
    #kMeans(clusterOptimum, x_pca,0,1)

def callGstoreData():
    print ("============= Gstore Data ===========================")
    train_df = load_df()     
    scaler = StandardScaler()
    scaler.fit(train_df)
    # Apply scalar transform 
    train_df1 = train_df
    train_df= scaler.transform(train_df)
    # call clusters
    clusterOptimum = 10
    findOptimumCluster(train_df)
    gmmEM(clusterOptimum, train_df, 0, 1)
    kMeans(clusterOptimum, train_df, 0, 1)

    # call dimension reduction n transfrom
    s=1000
    checkOptimaldimensionality(s)
    variancePCA(train_df1)
    nc = 5
    #x_pca = callPCA(train_df, 0.8, 2)
    #x_pca = callPCA(train_df, 0.9, 2)
    x_pca = callPCA(train_df, nc, 2)
    x_ica = callICA(train_df,nc,2)
    x_randomiProj = callRandomProjection(train_df,nc,2)
    varianceRandomizedPCA(train_df1)
    nc = 20
    x_randomizedPCA = callRandomizedPCA(train_df,nc,2)

    # call neural nw after dimension reduction
    callNeuralNwClassifier(train_df1,x_pca)

    # re-run clusters
    findOptimumCluster(x_pca)
    # 2nd time optimum cluster gives whatever value
    clusterOptimum = 8
    gmmEM(clusterOptimum, x_pca, 0, 1)
    kmeans = kMeans(clusterOptimum, x_pca, 0, 1)

    print ("2nd time clusters on ICA")
    gmmEM(clusterOptimum, x_ica, 0, 1)
    kmeans = kMeans(clusterOptimum, x_ica, 0, 1)

    print ("2nd time clusters on RP")
    gmmEM(clusterOptimum, x_randomiProj, 0, 1)
    kmeans = kMeans(clusterOptimum, x_randomiProj, 0, 1)

    print ("2nd time clusters on RPCA")
    gmmEM(clusterOptimum, x_randomizedPCA, 0, 1)
    kmeans = kMeans(clusterOptimum, x_randomizedPCA, 0, 1)
           
    # now cluster re-run, pass cluster to neural nw
    labels = kmeans.labels_
    print("kmeans labels shape at start", labels.shape)
    cp_train_df = train_df1
    cp_train_df['clusters'] = labels
    callNeuralNwClassifier(train_df1, cp_train_df)

def callImageData(): 
    mnist = fetch_mldata('MNIST original')   
    print ("============= Image Data ===========================")
    train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/5.0, random_state=0)
    scaler = StandardScaler()
    scaler.fit(train_img)
    train_img = scaler.transform(train_img)
     
    # run dimension reduction
    #s=56000
    # skip these methods as lot of time taken, machine hangs. Ckusters skipped too
    #checkOptimaldimensionality(s)
    #variancePCA(train_img)
    nc = 0.95 #95% of the variance will be retained by this flag
    x_pca = callPCA(train_img, nc, 3)
    
callEnergyData()
callGstoreData()
callImageData()
