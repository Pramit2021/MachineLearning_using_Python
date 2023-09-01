import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

def predict_using_sklearn():
    df = pd.read_excel('C:\\Users\\ASUS\\Desktop\\python files\\projects\\Machine_learning\\Gradientdescent\\Exercise\\test_score.xlsx') 
    r = LinearRegression()
    r.fit(df[['math']],df['cs'])
    return r.coef_,r.intercept_

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000000
    n = len(x)
    learning_rate = 0.0002
    rel_tot = 0
    cost_previous = 0

    for i in range(iterations):
        y_predicted = m_curr*x + b_curr
        Cost = (1/n) * sum([ val**2 for val in (y-y_predicted)])
        md = (-2/n)*sum(x*(y-y_predicted))
        bd = (-2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(Cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = Cost
        print("m {}, b {}, cost {}, iterations {}".format(m_curr,b_curr,Cost,i))

    return m_curr, b_curr

if __name__ == "__main__":
    df = pd.read_excel('C:\\Users\\ASUS\\Desktop\\python files\\projects\\Machine_learning\\Gradientdescent\\Exercise\\test_score.xlsx')
    x = np.array(df['math'])
    y = np.array(df['cs'])

    m,b = gradient_descent(x,y)
    print("Using gradient descent: Coeff {}, Intercept{}".format(m,b))

    m_sklearn, b_sklearn = predict_using_sklearn()
    print("Using sklearn: Coeff {}, Intercept {}".format(m_sklearn, b_sklearn))

