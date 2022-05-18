import numpy as np
import matplotlib.pyplot as plt

def main():

    x=np.array([10, 9, 2, 15, 10, 16, 11, 16])
    y=np.array([95, 80, 10, 50, 45, 98, 38, 93])

    b=estimated_coeff(x,y)

    print("Estimated Coefficients:\nb[0]={}\nb[1]={}".format(b[0],b[1]))

    plotting(x,y,b)

def estimated_coeff(x,y):

    n=np.size(x)

    m_x=np.mean(x)
    m_y=np.mean(y)

    ss_xy=np.sum(y*x)-n*m_x*m_y#cross derivation
    ss_xx=np.sum(x*x)-n*m_x*m_x#derivation of x

    b_1=ss_xy/ss_xx
    b_0=m_y - b_1 * m_x

    return (b_0,b_1)

def plotting(x,y,b):
    plt.scatter(x,y,color='m',marker='o',s=30)
    y_pred=b[0]+b[1]*x
    plt.plot(x,y_pred,color='g')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    main()