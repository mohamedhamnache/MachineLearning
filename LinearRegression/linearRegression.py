import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

#import the csv file
df = pd.read_csv('data.csv',delim_whitespace=True)

print(df.head())

x_train=df['Father'].values[:,np.newaxis]
y_train =df['Son'].values
lm=LinearRegression()
lm.fit(x_train,y_train)

x_test = [[72.8],[61.1],[67.4],[70.2],[75.6],[60.2],[65.3],[59.2]]

predictions = lm.predict(x_test)

print (predictions)

# plot the training data
plt.scatter(x_train,y_train,color='blue')
plt.plot(x_test,predictions,color='black',lineWidth=3)
plt.xlabel('Father height in inches')
plt.ylabel('Son height in inches')
plt.show()