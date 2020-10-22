import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
# this is used to split things into diff datasets
from matplotlib import pyplot as plt
from sklearn import metrics
# helps measure accuracy of model.

dataset = pandas.read_csv("regression_dataset.csv")
# print(dataset)

target = dataset.iloc[:,0].values
# print(target)

data = dataset.iloc[:,3:9].values
# print(data)

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size=0.25, random_state=0)
# our test dataset will be 1/4 of the full dataset. random_state=0 gives you same result every time.

print(data.shape)
print(target.shape)

print(data_training.shape)
print(data_test.shape)

print(target_training.shape)
print(target_test.shape)

machine = linear_model.LinearRegression()
machine.fit(data_training, target_training)

results = machine.predict(data_test)
# this is not actually predicting NEW data, but the data that we already have and are using as a test.
print(results)

plt.scatter(target_test, results)
plt.xlabel("target_test")
plt.ylabel("machine_predict")
# will need to save into pic file:
plt.savefig("scatter_test.png")
# each dot represents a row. 
# plot is helpful, but doesn't give us helpful exact info

print(metrics.r2_score(target_test,results))
# R2 is a measure of model fitness concerning how much of the variation in y is explained by x.
# it's between 0 and 1. 
# "explain" here means













