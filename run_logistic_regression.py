from sklearn import linear_model
import pandas
# pandas goes first bc we need to read csv
dataset = pandas.read_csv("regression_dataset.csv")
# dataset is actually a data frame.
# sk learn can learn some of data frame, but not all of it. so we use pandas.
# so we assume that sk learn can't read the df so we need to transform it into format that sk learn likes to read.
# print(dataset)
# target = dataset.iloc[:,0]
# [:, 2] is indicating all rows, for the 0th indexed column, y1
# target means y variable
target = dataset.iloc[:,1].values
# .values puts it into the format of a very long array. this is what we need for sklean
# "1" gets info from the column indexed as 1 as the output.
print(target)

data = dataset.iloc[:,3:9].values
# this gets values of all observations for the 3rd through 8th columns (x1-x6)
# data means x variable
# .values turns it into array of arrays, which sklearn can use.
# each row in this setup is an observation.
# depending on the dataset you need to change the numbers.
print(data)

# need to use a construtor to make a model. the constructor is LinearRegression() which is like BeautifulSoup in that it is a constructor.


# machine = linear_model.LinearRegression()
machine = linear_model.LogisticRegression()
machine.fit(data, target)
# this is fitting it. this line is there for pretty much every one of these programs in sklearn.
# print(machine)
# 
# the machine has already learned about the data now that we'ver run this.

# now we have a new person. we want the machine to predict the y.
new_data = [
	[-0.44,-0.29,0.51,0.92,-0.012,0],
	[1,-0.55,0.55,1.3,-0.011,1],
	[2,-0.53,0.9,0.36,-0.0123,0],
	[2.3,-0.23,0.33,0.32,-0.019,1],
	[10,-0.23,0.33,0.32,-0.019,1],
]
# the 2 [[]]s is because the new data needs to have the same format as the data that's already in the dataframe.
results = machine.predict(new_data)
print(results)
# with logistic regression, it doesn't give you probabilities. it gives you predictions of discretely whether or not something is true. so this makes a lot of sense to use in many cases.
# this is why we use LOGISTIC REGRESSIONS IN BINARY SITUATIONS!!!!!
# we can't know whether or not our model is working well/if our prediction is correct.


