import pandas as pd
import numpy as np
data=pd.read_excel('rawdata.xlsx', sheet_name=0,header=0,index_col=0)
m,n=data.shape  #获取行数m和列数n
#熵权法计算
def Y_ij(data1):   #矩阵标准化(min-max标准化)
    for i in data1.columns:
       for j in range(n+1):
           if i == str(f'X{j}负'):  #负向指标
               data1[i]=(np.max(data1[i])-data1[i])/(np.max(data1[i])-np.min(data1[i]))
           else:   #正向指标
               data1[i]=(data1[i]-np.min(data1[i]))/(np.max(data1[i])-np.min(data1[i]))
    return data1
Y_ij=Y_ij(data)  #标准化矩阵
None_ij = [[None] * n for i in range(m)]  #新建空矩阵
def E_j(data3):  #计算熵值
    data3 = np.array(data3)
    E = np.array(None_ij)
    for i in range(m):
        for j in range(n):
            if data3[i][j] == 0:
                e_ij = 0.0
            else:
                P_ij = data3[i][j] / data3.sum(axis=0)[j]  #计算比重
                e_ij = (-1 / np.log(m)) * P_ij * np.log(P_ij)
            E[i][j] = e_ij
    E_j=E.sum(axis=0)
    return E_j
E_j = E_j(Y_ij)  #熵值
G_j = 1 - E_j    #计算差异系数
W_j = G_j / sum(G_j)   #计算权重
WW= pd.Series(W_j, index=data.columns, name='指标权重')
#print(WW)
Y_ij.to_excel("Y_ij.xlsx",sheet_name='Y_ij')
WW.to_excel("WW.xlsx",sheet_name='WW',index=False)

# transpose x,y and print the shape
wdf = pd.read_excel('WW.xlsx').transpose()
print(wdf)
weightArray = wdf.values
print(wdf.shape[1],wdf.shape[0])
