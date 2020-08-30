import pandas as pd

def compute(x):
    return 2*x*x+1
x=[i for i in range(100)]
y=[compute(i) for i in x]
data = pd.DataFrame({'x':x,'y':y})
# 查看pearson系数
print(data.corr())
print(data.corr(method='spearman'))
print(data.corr(method='kendall'))
#以下为结果部分
          x         y
x  1.000000  0.967644
y  0.967644  1.000000
     x    y
x  1.0  1.0
y  1.0  1.0
     x    y
x  1.0  1.0
y  1.0  1.0