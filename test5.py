def loss_function(k, b, x, y):
    sum = 0
    for i in range(len(x)):
        err = (k * x[i] + b) - y[i]
        sum += err ** 2
    J = sum / len(x)
    return J


def sum1(k, b, x, y):
    sum = 0
    for i in range(len(x)):
        sum += ((k * x[i] + b) - y[i])
    return sum


def sum2(k, b, x, y):
    sum = 0
    for i in range(len(x)):
        sum += ((k * x[i] + b) - y[i]) * x[i]
    return sum


def gradient_descent(k, b, alfa, N, x, y):
    for j in range(N):
        k = k - alfa * 2 / len(x) * sum2(k, b, x, y)
        b = b - alfa * 2 / len(x) * sum1(k, b, x, y)
        print(loss_function(k, b, x, y))
    return k, b


x = [1, 1, 2, 3, 4, 3, 4, 6, 4]
y = [2, 1, 0.5, 1, 3, 3, 2, 5, 4]
a = loss_function(0, 0, x, y)
print(a)

v = gradient_descent(0, 0, 0.1, 10, x, y)

