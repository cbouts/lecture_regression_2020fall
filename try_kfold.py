import pandas
from sklearn import linear_model 
from sklearn import metrics
from sklearn.model_selection import KFold

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix


dataset = pandas.read_csv("regression_dataset.csv")

target = dataset.iloc[:,0].values
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
	machine = linear_model.LinearRegression()
	machine.fit(data_training, target_training)
	results = machine.predict(data_test)
	print(metrics.r2_score(target_test, results))












