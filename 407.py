import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

dataset = pd.read_csv("Salary_Data.csv")
#dataset = pd.read_csv("teachers_salaries.csv")


describe = dataset.describe() #summary statistics for numerical columns
print("describe: \n\n",describe)
print("\n\n")
max = dataset.max() # Returns the highest value in each column
print("max value: \n",max)
min = dataset.min() # Returns the lowest value in each column
print("\nmin value: \n",min)



X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#seaborn visualising

sns.pairplot(dataset)

sns.lmplot(x='YearsExperience',y='Salary',data=dataset)
sns.heatmap(dataset.corr(),cmap = 'Blues', annot=True)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)
         

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)
accuracies = cross_val_score(regressor, X = X_train, y = y_train, cv = 10)

score = -1

print("Mean: %",score*accuracies.mean()*100)
print("std: %",accuracies.std()*100)

            
            
y_predict = regressor.predict(X_test)


from sklearn import metrics


#mean absolute error
mae = metrics.mean_absolute_error(y_test,y_predict)
print("mean absolute error: ",mae)
#mean squared error
mse = metrics.mean_squared_error(y_test, y_predict)
print("mean squared error: ",mse)
#root mean square error
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_predict))
print("root mean square error: ",rmse)

print("coef: ",regressor.coef_)




# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


