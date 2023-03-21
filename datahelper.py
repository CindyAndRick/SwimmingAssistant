import pandas as pd
import numpy as np

# 数据合并

df1 = pd.read_csv('./data/feature/ordinary.csv', index_col=0)
df2 = pd.read_csv('./data/feature/breath.csv', index_col=0)
df3 = pd.read_csv('./data/feature/swing.csv', index_col=0)
df4 = pd.read_csv('./data/feature/upp.csv', index_col=0)
df5 = pd.read_csv('./data/feature/out.csv', index_col=0)

df = pd.concat([df1, df2], ignore_index=True)
df = pd.concat([df, df3], ignore_index=True)
df = pd.concat([df, df4], ignore_index=True)
df = pd.concat([df, df5], ignore_index=True)


# 拆分

df.to_csv('./data/feature/data.csv')

df = pd.read_csv('./data/feature/data.csv', index_col=0)
df = df.sample(frac=1.0)
df = df.reset_index(drop=True)
# print(df)
df1 = df.iloc[0:210]
df2 = df.iloc[210:]
# print(df1)
# print(df2)
df1.to_csv('./data/feature/test.csv')
df2.to_csv('./data/feature/train.csv')

# 标准化

x_train=pd.read_csv('./data/feature/train.csv').iloc[:,1:]

# print(x_train)

y_train = x_train.iloc[:,-1]

# print(y_train)

x_train = x_train.iloc[:,:-1]

# print(x_train)

x_train_normal = x_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

# print(x_train_normal)

y_train.to_csv('./data/feature/y_train.csv')
x_train_normal.to_csv('./data/feature/x_train_norm.csv')

x_test=pd.read_csv('./data/feature/test.csv').iloc[:,1:]

# print(x_test)

y_test = x_test.iloc[:,-1]

# print(y_test)

x_test = x_test.iloc[:,:-1]

# print(x_test)

x_test_normal = x_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

# print(x_test_normal)

y_test.to_csv('./data/feature/y_test.csv')
x_test_normal.to_csv('./data/feature/x_test_norm.csv')