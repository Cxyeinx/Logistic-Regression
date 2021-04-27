import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
style.use("ggplot")


df = pd.read_csv("Student-Pass-Fail-Data.csv")


df.head()


x = np.array(df["Self_Study_Daily"])
x = np.expand_dims(x, axis=-1)
y = np.array(df["Pass_Or_Fail"])
y= np.expand_dims(y, axis=-1)
print(x.shape, y.shape)


plt.scatter(x, y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


model = LogisticRegression()
model.fit(x_train, y_train)


pred = model.predict(x_test)


plt.scatter(x_test, pred)
plt.title("Predicted")


plt.scatter(x_test, y_test)
plt.title("Original")



