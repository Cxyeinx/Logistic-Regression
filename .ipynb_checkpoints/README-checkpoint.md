# Logistic Regression

In Statistics, Logistic Model is the model used to define probability of a class as pass/fail or win/loose.

Logistic Regression is alot similar to Linear Regression.

The formula of Linear Regression was 
$$
y = mx + b
$$
where $m$ represents the slope and $b$ represents the y-intercepts.

If we ignore the y-intercepts, it becomes
$$
y = mx
$$

In Statistics this is written as 
$$
h = \theta X
$$
where $h$ is the predicted value and $X$ is the input values. $\theta$ is initialised randomly and later improved.

For Logistic Regression, we need the value to be b/w 0 and 1. \
So as we use the Logistic Function or the Sigmoid function. 

Sigmoid function can be defined as 
$$
f(x) = \frac{1}{1 + e^{-x}}
$$

Substituting the previous formula with sigmoid function, we get
$$
h = \frac{1}{1 + e^{-z}}
$$

where $z$ is the input data, multiplied with randomly initialised value theta. I can be shown by $z = \theta X$

Now lets see at the cost function of linear regression. 
$$
C(h, y) = \frac{1}{2} (h - y)^2
$$
and the average cost function orloss function was 
$$
J = \frac{1}{n} \sum^{n}_{i=0}\frac{1}{2} (h-y)^2
$$
where, $n$ is the number of training data, $h$ is the predicted value, $y$ is the original value. 1/2 for optimization.

This is for Linear Regression, but Logistic Regression uses Sigmoid Function which is not linear. 

So the simplified cost function would be
$$
J = \frac{1}{n} \sum^{n}_{i=0} y \cdot log(h) + (1 - y) \cdot log(1 - h)
$$

Time for developing the gradient descent for initializing the $\theta$ value.

$$
\theta = \theta - \alpha \sum^{n}_{i=0} (h - y)x_j
$$
where $\alpha$ is the learning rate.


So there we go, the Logistic Regression from scratch is done :P