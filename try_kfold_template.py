import kfold_template
	# this is the template file but we don't need the .py here
from sklearn import linear_model
import pandas



dataset = pandas.read_csv("regression_dataset.csv")

target = dataset.iloc[:,2].values
data = dataset.iloc[:,3:9].values

machine = linear_model.LogisticRegression()
results_accuracy, results_confusion = kfold_template.run_kfold(4,data,target,machine)
# kfold_template tells you where the function comes from
print(results_accuracy)
for i in results_confusion:
	print(i)