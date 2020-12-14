'''
|**********************************************************************;
* Project           : Experida Hotel Recommendations
*
* Program name      : owens_MLAfinal.py
*
* Author            : Jessie Ann Owens
*
* Date created      : 10/10/20
*
* Purpose           : Final project for Introduction to Machine Learning
*					Aplications. Predict customer cluster based on 
*					search engine events. 
*
|**********************************************************************;
'''

import pandas as pd
import numpy as np
import random
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


def generatesample_data(n_pop,n_sample,file):
	"""Generates sample and write to csv

    Parameters
    ----------
    n_pop : int
        Number of events in file
    n_sample : int
        size of sample
    file : str
    	name of data file

    Returns
    -------
    dataframe
        a pandas dataframe of sample data
    """
	print("reading...")
	skip = sorted(random.sample(range(n_pop),n_pop - n_sample))
	df = pd.read_csv(file,skiprows=skip)
	print(df.head(10))
	header = ['date_time', 'site_name', 'posa_continent', 'user_location_country',
	       'user_location_region', 'user_location_city',
	       'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
	       'channel', 'srch_ci', 'srch_co', 'srch_adults_cnt', 'srch_children_cnt',
	       'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id',
	       'is_booking', 'cnt', 'hotel_continent', 'hotel_country', 'hotel_market',
	       'hotel_cluster']
	df.columns = header
	df.to_csv('feat_sample.csv',columns = header)
	return df

def read_sample(file,prepped):
	"""Reads file 

    Parameters
    ----------
    file : str
        .csv file with sample data 
    prepped: int
        flag variable, if 1 then data has been prepped
        and column headers are different

    Returns
    -------
    dataframe
        a pandas dataframe of sample data
    """
	print("Reading sample...")
	df = pd.read_csv(file)
	if not prepped:
		c = ['n','date_time', 'site_name', 'posa_continent', 'user_location_country',
		       'user_location_region', 'user_location_city',
		       'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
		       'channel', 'srch_ci', 'srch_co', 'srch_adults_cnt', 'srch_children_cnt',
		       'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id',
		       'is_booking', 'cnt', 'hotel_continent', 'hotel_country', 'hotel_market','hotel_cluster'] 
		print(len(c))
		df.columns = c
	else:
		df.columns = ['n','site_name', 'posa_continent', 'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
       'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
       'srch_destination_id', 'srch_destination_type_id', 'is_booking', 'cnt',
       'hotel_continent', 'hotel_country', 'hotel_market', 'duration_of_stay',
       'hotel_cluster', 'most_pop_clust']

	print("Done reading")
	return df

def most_popular(sid,hclust):
	"""calculates most popular
	hotel cluster for a given search location 

    Parameters
    ----------
    sid : pd.Series
        search_destination_ids
    hcluster: pd.Series
        hotel_cluster corresponding to the corresponding destination

    Returns
    -------
    dataframe
        a pandas dataframe with a unique list of search_destination_ids
        and the associated most popular hotel cluster
    """
	print("Finding most popular...")
	counts = np.zeros((sid.max()+1,100))
	mp = np.zeros((sid.max()+1,2),dtype = 'int64')
	for i in range(len(sid)): #accumulate counts
		counts[sid.iloc[i],hclust.iloc[i]] += 1

	for i in range(counts.shape[0]): #assigns most popular cluster
		mp[i,0] = i
		mp[i,1] = np.argmax(counts[i])

	df = pd.DataFrame(mp,columns = ['sid','most_pop_clust'])

	return df

def prep_data(df):
	"""Performs initial processing of data 
    Parameters
    ----------
   df : pd.DataFrame
   		data

    Returns
    -------
    dataframe
        a pandas dataframe with cleaned data
    """
	print("Prepping Data...")

	#calculating date difference between checkin time and checkout time
	times = (pd.to_datetime(df['srch_co'],errors = 'coerce') - pd.to_datetime(df['srch_ci'], errors= 'coerce')).astype('timedelta64[D]')
	df = df.drop(columns = ['date_time','srch_co','srch_ci'])
	df.insert(len(df.columns)-1,'duration_of_stay',times)

	#replacing na values
	df = df.dropna(axis = 0, subset=['hotel_cluster']) #don't care about unlabled rows
	df['orig_destination_distance'].fillna((df['orig_destination_distance'].mean()),inplace=True)
	df['duration_of_stay'].fillna((df['duration_of_stay'].mean()),inplace=True)

	#calculating most popular cluster for hotel region
	mp = most_popular(df['srch_destination_id'],df['hotel_cluster'])
	df = df.join(mp,on = ['srch_destination_id'])
	df = df.drop(columns=['sid'])

	print(df.columns)
	c = ['site_name', 'posa_continent', 'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
       'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
       'srch_destination_id', 'srch_destination_type_id', 'is_booking', 'cnt',
       'hotel_continent', 'hotel_country', 'hotel_market', 'duration_of_stay',
       'hotel_cluster', 'most_pop_clust']
	df.to_csv('prepped_train1.csv',columns = c)

	return df

def eda(df):
	"""Performs exploratory analysis of data. Analysis includes
	basic visualizations and calculations for the purpose of understanding
	the data
    Parameters
    ----------
    df : pd.DataFrame
   		data

    Returns
    -------
    None
    """

	g1 = sns.countplot(x = 'hotel_cluster',data=df,color = "dimgray")
	g1.set(xticklabels=[])
	plt.show()

	g1 = sns.countplot(x = 'most_pop_clust',data=df, color = "dimgray")
	g1.set(xticklabels=[])
	plt.show()

	x = df['duration_of_stay']
	g1 = sns.boxplot(data = x,showfliers = False, color = "dimgray")
	g1.set(xlabel = '')
	plt.show()


	x = df['orig_destination_distance']
	sns.boxplot(data = x)
	plt.show()

	sns.countplot(x = 'channel',data =df)
	plt.show()

	matrix = np.triu(df.corr())
	sns.heatmap(df.corr(), linewidths=.5, cmap ='Greys', mask=matrix )
	plt.show()

	df1 = df.drop(columns=['n','date_time'])
	var = df.var(axis=0,skipna=True)
	df_var = var.to_frame()

	# columns = ['site_name', 'posa_continent', 'user_location_country',
 #       'user_location_region', 'user_location_city',
 #       'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
 #       'channel', 'srch_adults_cnt', 'srch_children_cnt',
 #       'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id',
 #       'is_booking', 'cnt', 'hotel_continent', 'hotel_country', 'hotel_market','duration_of_stay',
 #       'hotel_cluster']

	
	#df_var.insert(0,'column_name',columns)
	df_var.iloc[:,0] = df_var.iloc[:,0].astype('int64')
	print(df_var.round(2))



	return 

def separate_classes(y_true,y_pred):
	"""Groups predictions by cluster
    Parameters - not called during program, used in attempt
    to find mAP
    ----------
   y_true : pd.Series
   		ground truth
   	y_pred : np.ndarray
   		predicted values

    Returns
    -------
    dictionary
        keys are class labels, values are the ground truth and predict class
        for a given user event
    """
	print("separating classes ... ")
	by_class = {new_list: {'y_true':[],'y_pred':[]} for new_list in range(100)}

	for i in range(len(y_pred)):
		t = y_true.iloc[i]
		by_class[t]['y_true'].append(t)
		by_class[t]['y_pred'].append(y_pred[i])

	return by_class

def map1(y_true,y_pred):
	"""Calculates mean average precision for multiclass single label
    ----------
   y_true : pd.Series
   		ground truth
   	y_pred : np.ndarray
   		predicted values

    Returns
    -------
    float
        mean average precision
    """

	# by_class = separate_classes(y_true,y_pred)
	# print("Calculating average precision accross classes...")

	# avg_precs = []
	# for i in range(100):
	# 	y_t =np.array(by_class[i]['y_true'])
	# 	y_p = np.array(by_class[i]['y_pred'])
	# 	print(type(y_t),type(y_p))
	# 	avg_precs.append(metrics.average_precision_score(y_t,y_p))

	# return sum(avg_precs)/100
	#seperating 
	print(metrics.precision_score(y_true,y_pred,average='micro'))

def map5(y_true,y_pred,n,U):
	"""Calculates mean average precision at cutoff five for multiclass single label
    ----------
   y_true : pd.Series
   		ground truth
   	y_pred : np.ndarray
   		predicted values

    Returns
    -------
    float
        mean average precision at cutoff five
    """
	print("Calculating MAP@5")
	temp = 0
	for u in range(U):
		for k in range(min(5,n)):
			temp += metrics.precision_score(y_true[u],y_pred[u,k],average='micro')
	return temp / U

def feature_selection(df,sample):
	"""runs feature selection algorithm to calculate
	feature importance

    Parameters
    ----------
    df : pd.DataFrame
        data
    sample : int
        flag variable, if whole datset take a sample

    Returns
    -------
    None
    """
	if not sample:
		#shuffle
		df = df.sample(frac=1)
		#df.head(df.shape[0] *80)
		df = df.head(500)

	y = df['hotel_cluster']
	X = df.drop(columns= ['hotel_cluster'])

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=99)

	estimator = RandomForestClassifier(random_state	=99, max_depth = 10)
	print("Fitting RF Classifier for Feature Selection ... ")
	estimator.fit(X_train,y_train)

	selector = RFECV(estimator,cv = 10, step = .50)
	print("Fitting feature selector ...")
	selector = selector.fit(X_train,y_train	)

	print("FIT!")
	mask = selector.get_support() #list of booleans
	features = [] 
	for b, feature in zip(mask, X_train.columns):
 		if b:
 			features.append(feature)

	print("Num feat: {}".format(selector.n_features_))
	print("Features: {}".format(features))

	plt.barh(range(X_train.shape[1]), estimator.feature_importances_, align='center') 
	plt.yticks(np.arange(X_train.shape[1]), X_train.columns.values) 
	plt.xlabel('Feature importance')
	plt.ylabel('Feature')
	plt.show()

def hyper_parameter_tuning(X_train,X_test,y_train,y_test,model_no):
	"""Finds the best parameters for a classification model

    Parameters
    ----------
    X_train : pd.DataFrame
        data containing independent varaibles, used to train
        classifier
    X_test : pd.DataFrame
       data containing independent variables, used the test
       the performance of the classifier
    y_train : pd.Series
    	labels of user events, used to train classifier
    y_test : pd.Series
    	labels of user events, used to test the performance
    	of the classifier

    Returns
    -------
    None
    """
	if model_no == 1:
		gsc = GridSearchCV(
			estimator = RandomForestClassifier(random_state = 99),
			param_grid = {
				'n_estimators': [50,100,150],
				'criterion': ["gini","entropy"],
				'max_depth':[5,10,25,50]
			},
			scoring = "accuracy"
		)
		print("Hyperparameter tuning...")
		grid = gsc.fit(X_train,y_train)
		best_params = grid.best_params_
		print(best_params)

	if model_no == 2:
		gsc = GridSearchCV(
			estimator = KNeighborsClassifier(random_state = 99),
			param_grid	= {
					'n_neighbors': [1,2,4,10],
					'algorithm' : ["auto","ball_tree","kd_tree"],
					'leaf_size' : [20,30,50,100,125],
					'metric' : ["minkowski","euclidean","manhattan","chebyshev"] 
			},
			scoring = "accuracy"
		) 
		print("Hyperparameter tuning...")
		grid = gsc.fit(X_train,y_train)
		best_params = grid.best_params_
		print(best_params)
	
def modelNo0(X_train, X_test, y_train, y_test): 
	"""Runs Logistic Regression to classify 

    Parameters
    ----------
    X_train : pd.DataFrame
        data containing independent varaibles, used to train
        classifier
    X_test : pd.DataFrame
       data containing independent variables, used the test
       the performance of the classifier
    y_train : pd.Series
    	labels of user events, used to train classifier
    y_test : pd.Series
    	labels of user events, used to test the performance
    	of the classifier

    Returns
    -------
    float
    	mean average precision 
    """
	classifier = LogisticRegression(random_state=99,max_iter=1000,multi_class='multinomial',solver='newton-cg')
	classifier.fit(X_train,y_train)
	print("predicting...")
	y_pred = classifier.predict(X_test)

	print(map1(y_test,y_pred))
	return map1(y_test,y_pred)

def modelNo1(X_train,X_test,y_train,y_test):  
	"""Runs Random Forest Classifier

    Parameters
    ----------
    X_train : pd.DataFrame
        data containing independent varaibles, used to train
        classifier
    X_test : pd.DataFrame
       data containing independent variables, used the test
       the performance of the classifier
    y_train : pd.Series
    	labels of user events, used to train classifier
    y_test : pd.Series
    	labels of user events, used to test the performance
    	of the classifier

    Returns
    -------
    float
    	mean average precision 
    """
	classifier = RandomForestClassifier(n_estimators = 50, criterion = "entropy", max_depth = 10, random_state = 99)
	classifier.fit(X_train,y_train)
	print("predicting...")
	y_pred = classifier.predict(X_test)
	print(y_pred)
	print(map1(y_test,y_pred))

	return

def modelNo2(X_train,X_test,y_train,y_test): 
	"""Runs K Nearest Neighbors Classifier

    Parameters
    ----------
    X_train : pd.DataFrame
        data containing independent varaibles, used to train
        classifier
    X_test : pd.DataFrame
       data containing independent variables, used the test
       the performance of the classifier
    y_train : pd.Series
    	labels of user events, used to train classifier
    y_test : pd.Series
    	labels of user events, used to test the performance
    	of the classifier

    Returns
    -------
    float
    	mean average precision 
    """
	classifier = KNeighborsClassifier(n_neighbors = 100, algorithm = "kd_tree",leaf_size = 100)
	classifier.fit(X_train,y_train)
	print("predicting...")
	y_pred = classifier.predict(X_test)
	m = map1(y_test,y_pred)

	return m

def modelNo3(D_train,D_test,y_test,eta,gamma,max_depth,alpha,steps):
	"""Runs XGBoost CLassifier

    Parameters
    ----------
    X_train : pd.DataFrame
        data containing independent varaibles, used to train
        classifier
    X_test : pd.DataFrame
       data containing independent variables, used the test
       the performance of the classifier
    y_train : pd.Series
    	labels of user events, used to train classifier
    y_test : pd.Series
    	labels of user events, used to test the performance
    	of the classifier
	eta : float
		learning rate
	gamma : int
		minumum loss reduction
	max_depth : int
		maximum depth of the tree
	alpha : int
		regulariazation term
	steps : int
		number of steps

    Returns
    -------
    float
    	mean average precision 
    """
	param = {
		'eta':eta,
		'gamma' : gamma,
		'max_depth': max_depth,
		'alpha' : alpha,
		'num_class': 100 }
	steps = steps

	print("Training data..")
	model = xgb.train(param,D_train,steps)
	print("Predicting Data...")
	preds = model.predict(D_test)

	temp = y_test.to_numpy()
	preds = preds.astype('int64')
	print(map1(temp,preds))
	return

def driver(df,modNo):
	"""Splits data and calls classification models

    Parameters
    ----------
    df : pd.DataFrame
        data
    modNo : int
    	indicates what model number to run

    Returns
    -------
    float
    	mean average precision 
    """
	y = df['hotel_cluster']
	X = df[['most_pop_clust','hotel_market','hotel_continent','hotel_country','orig_destination_distance']]
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.30,random_state=99)
	if modNo == 1:
		#hyper_parameter_tuning(X_train,X_test,y_train,y_test,2)
		print("Running RandomForestClassifier...")
		modelNo1(X_train,X_test,y_train,y_test)
	elif modNo==2:
		print("Running KNN Classifier...")
		#hyper_parameter_tuning(X_train,X_test,y_train,y_test,2)
		modelNo2(X_train,X_test,y_train,y_test)

	else:
		D_train = xgb.DMatrix(X_train,label=y_train)
		D_test = xgb.DMatrix(X_test,y_test)
		
		#modelNo3(D_train,D_test,y_test,0.30,10,10,1,10)
		modelNo3(D_train,D_test,y_test,0.30,10,10,1,20)
		#modelNo3(D_train,D_test,y_test,0.30,10,10,1,50)
		#modelNo3(D_train,D_test,y_test,0.30,10,10,1,100)


if __name__ == '__main__':
	n_pop = int(37.7 * 1000000)
	n_sample = 3000000
	#file = 'prepped_train1.csv'
	#file = 'sample2.csv'
	# #file = 'train.csv'
	file = "ASSIGN FILE HERE"
	print("Reading Data...")
	# #df = generatesample_data(n_pop,n_sample,file)
	df = read_sample(file,1)
	print(df.shape)
	df = df.sample(frac=1)
	df = df.head(500000)

	df = prep_data(df)
	feature_selection(df,1)
	eda(df)
	

	driver(df,1)
	driver(df,2)
	driver(df,3)



