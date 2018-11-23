import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')

def hypothesis(x, k, b):
    return k * x + b


def loss_function(k, b, x, y):
    return sum([(y[i] - hypothesis(x[i], k, b)) ** 2 for i in range(len(x))]) / len(x)


def gradient_descent(k, b, alfa, N, x, y):
    for j in range(N):
        loss = loss_function(k, b, x, y)
        k_ = k
        k += alfa / len(x) * sum([(y[i] - hypothesis(x[i], k, b)) * x[i] for i in range(len(x))])
        b += alfa / len(x) * sum([(y[i] - hypothesis(x[i], k_, b)) for i in range(len(x))])
        loss_list.append((j, loss - loss_function(k, b, x, y)))
        theta_list.append((k, b))
    return k, b


x = [1, 1, 2, 3, 4, 3, 4, 6, 4]
y = [2, 1, 0.5, 1, 3, 3, 2, 5, 4]

alfa = 0.1
N = 10
loss_list = []
theta_list = [(0, 0)] # начальные значения задаются здесь
theta = gradient_descent(theta_list[0][0], theta_list[0][1], alfa, N, x, y)

# функция регрессии
y_ = []
for i in range(len(x)):
    y_.append(hypothesis(x[i], theta[0], theta[1]))

# график линейной регрессии
plt.plot(x, y_, 'r', label=('Hypothesis function {:.3f} * x + {:.3f}'.format(theta[0], theta[1])))
plt.plot(x, y, 'bs', label=('Original data'), markersize=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear regression by gradient descent: alfa = {:.2f}, N = {:.0f}'.format(alfa, N))
plt.show()

# график ошибки
plt.xticks(np.arange(0, N, 1), np.arange(1, N + 1, 1), )
plt.plot(*zip(*loss_list), 'go', markersize=5)
plt.xlabel('Number of iteration')
plt.ylabel('Loss')
plt.title('Loss by iteration: alfa = {:.2f}, N = {:.0f}'.format(alfa, N))
#plt.grid()
plt.show()

print(*theta_list)
print(*loss_list)
print(*theta)
