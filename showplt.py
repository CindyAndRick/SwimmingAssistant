import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("./data/extract/swing.csv")


# print(data)

X = [i / 5 for i in range(0,len(data))]

Y0 = data['0'].tolist()
Y1 = data['1'].tolist()
Y2 = data['2'].tolist()
Y3 = data['3'].tolist()
Y4 = data['4'].tolist()
Y5 = data['5'].tolist()
# print(X)
# print(Y1)

plt.plot(X, Y0,label='ax')
plt.plot(X, Y1,label='ay')
plt.plot(X, Y2,label='az')
plt.plot(X, Y3,label='wx')
plt.plot(X, Y4,label='wy')
plt.plot(X, Y5,label='wz')

plt.xlabel('time(s)', fontsize=14)
plt.ylabel('data', fontsize=14)
plt.legend()
plt.show()