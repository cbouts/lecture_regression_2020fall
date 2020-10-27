import pandas
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def run_kfold(split_number,data,target,machine):
	kfold_object = KFold(n_splits=split_number)
	# split_number tells how many times you wana split the data
	# now we fit it with the data:
	kfold_object.get_n_splits(data)

	results_accuracy = []
	results_confusion = []

	for training_index, test_index in kfold_object.split(data):
		data_training, data_test = data[training_index], data[test_index]
		target_training, target_test = target[training_index], target[test_index]
		machine.fit(data_training, target_training)
		results = machine.predict(data_test)
		results_accuracy.append(metrics.accuracy_score(target_test, results))
		results_confusion.append(metrics.accuracy_score(target_test, results))
	return results_accuracy, results_confusion

	# run_kfold()
	# # but if we do this, we can't import it.
	 # so we put it in a thing like this:

# if __name__=='__main__' 

		# print(metrics.accuracy_score(target_test, results))
		# # accuracy score is for category variables
		# # r2 is for continuous variables.-- print(metrics.r2_score(target_test, results))
		# print(metrics.confusion_matrix(target_test, results))
		# # give you array of array. if your model has 5 different outcome variables y, you have a 5x5 matrix
		# # test case is on the vertical, what the model says is on the y axis.
		# # it gives you the amount of times the model says the thing on the horiszontal axis when the test case says the thing on the vertical axis

		# # usually we don't want to just print acuracy score and conf matrix.




