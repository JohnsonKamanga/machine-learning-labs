from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas

#create linear regression object
model_0 = linear_model.LinearRegression()

#open data set
data = pandas.read_csv('../datasets/Advertising.csv')

#get sales data
X = data['TV'].values.reshape(-1, 1)
Y = data['Sales'].values.reshape(-1, 1)

#create train split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#fit the data
model_0.fit(X_train, y_train)

# Create the plot
plt.plot(X_test, y_test, 'o', label='Actual')
plt.plot(X_test, model_0.predict(X_test), 'o', label='Predicted')

# Add a title and labels (optional)
plt.title("Simple Line Plot")
plt.xlabel("X Axis Label")
plt.ylabel("Y Axis Label")

plt.legend()

# Display the plot
plt.show()