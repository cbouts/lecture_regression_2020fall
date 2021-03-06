# logistic is good for category or dummy variables, not for continuous variables!! linear is good for that.
# regression_dataset.csv has different kinds of variables. 

import pandas
from sklearn import linear_model 
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dataset = pandas.read_csv("regression_dataset.csv")

# target = dataset.iloc[:,1].values
target = dataset.iloc[:,2].values
	# using y3
data = dataset.iloc[:,3:9].values

kfold_object = KFold(n_splits = 4)
kfold_object.get_n_splits(data)

# print(kfold_object)
# this contains the indexes that guide your dataset- what part's training and what part is test.
# indexes allow us to do for loop.

i = 0
for training_index, test_index in kfold_object.split(data):
	i = i +1
	print("round: ", i)
	print("training: ", training_index)
	print("test: ", test_index)
	# this makes the kfold groups
	data_training, data_test = data[training_index], data[test_index]
	target_training, target_test = target[training_index], target[test_index]
	machine = linear_model.LogisticRegression()
	machine.fit(data_training, target_training)
	results = machine.predict(data_test)
	print(metrics.accuracy_score(target_test, results))
		# accuracy score is for category variables
		# r2 is for continuous variables.-- print(metrics.r2_score(target_test, results))
	print(metrics.confusion_matrix(target_test, results))
		# give you array of array. it's 5x5. if your model has 5 different outcome variables y, you have a 5x5 matrix
		# test case is on the vertical, what the model says is on the y axis.
		# it gives you the amount of times the model says the thing on the horiszontal axis when the test case says the thing on the vertical axis









