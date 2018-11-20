import numpy as np

def loss_function(k, b, x, y):
    sum = 0
    for i in range(len(x)):
        err = (k * x[i] + b) - y[i]
        sum += err ** 2
    J = sum / len(x)
    return J


def hypothesis(x, theta0, theta1):
    return theta0 + theta1*x


def cost_func(theta0, theta1):
    theta0 = np.atleast_3d(np.asarray(theta0))
    theta1 = np.atleast_3d(np.asarray(theta1))
    return np.average((y-hypothesis(x, theta0, theta1))**2, axis=2)


x = np.array([1, 1, 2, 3, 4, 3, 4, 6, 4])
y = np.array([2, 1, 0.5, 1, 3, 3, 2, 5, 4])

N = 10
alpha = 0.05
theta = [np.array((0,0))]
J = [cost_func(*theta[0])[0]]
m = len(x)

for j in range(N-1):
    last_theta = theta[-1]
    this_theta = np.empty((2,))
    this_theta[0] = last_theta[0] - alpha / m * np.sum((hypothesis(x, *last_theta) - y))
    this_theta[1] = last_theta[1] - alpha / m * np.sum((hypothesis(x, *last_theta) - y) * x)
    theta.append(this_theta)
    J.append(cost_func(*this_theta))

print(J)