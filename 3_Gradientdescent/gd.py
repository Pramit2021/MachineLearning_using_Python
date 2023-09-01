import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learningrate = 0.1

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        Cost = (1/n) * sum([ val**2 for val in (y-y_predicted)])
        md = (-2/n)*sum(x*(y-y_predicted))
        bd = (-2/n)*sum(y-y_predicted)
        m_curr = m_curr - learningrate * md
        b_curr = b_curr - learningrate * bd
        print("m {}, b {}, Cost {}, iteration {}".format(m_curr,b_curr,Cost,i))

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x,y)