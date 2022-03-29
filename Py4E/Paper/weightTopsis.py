import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
df= pd.read_excel('rawdata.xlsx',sheet_name='raw') #read raw data
# delete index column
df1 = df.drop(labels='a',axis=1) #
# change to array values 
array = df1.values #transfer to array for further processing

#import weight data
wdf = pd.read_excel('WW.xlsx') # read weight dataframe
weightArray = np.array(wdf) # tranfer weight values
print('==',wdf.shape[1],wdf.shape[0]) 

#df.shape[0] is the number of rows

# normalize 
dataNorm = Normalizer(norm='l2').fit(array) # use l2 normalizer mathod ,
# transform to
dataNormed = dataNorm.transform(array)

normdf = pd.DataFrame(dataNormed)
print('===',normdf.shape[1],normdf.shape[0])
normdf.to_excel('normdata.xlsx') # export excel normalized data with weight
print('====',len(dataNormed))
print('=====',dataNormed.shape[0])
print('===**==',dataNormed.shape[1])
print('===**==',dataNormed[2,1],weightArray[2])


for i in range(dataNormed.shape[0]):
    for j in range(dataNormed.shape[1]):
        dataNormed[i,j] *= weightArray[j]

norm_weigh_df = pd.DataFrame(dataNormed)
norm_weigh_df.to_excel('normdweightata.xlsx')

# bestALternatives,and worstAlternatives 
bestAlternatives = np.zeros(20)
worstAlternatives = np.zeros(20)
for i in range(norm_weigh_df.shape[1]):
    bestAlternatives[i] = max(dataNormed[:,i])
    worstAlternatives[i] = min(dataNormed[:,i])

print(bestAlternatives[i])
print(worstAlternatives[i])

# calculate L2 distance 
worst_distance = np.zeros(8)
best_distance = np.zeros(8)
worst_distance_mat = np.copy(norm_weigh_df)
best_distance_mat = np.copy(norm_weigh_df)
for i in range(norm_weigh_df.shape[0]):
    for j in range(norm_weigh_df.shape[1]):
        worst_distance_mat[i][j] = (dataNormed[i][j]-worstAlternatives[j])**2
        best_distance_mat[i][j] = (dataNormed[i][j]-bestAlternatives[j])**2
        worst_distance[i] += worst_distance_mat[i][j]
        best_distance[i] += best_distance_mat[i][j]
for i in range(norm_weigh_df.shape[0]):
    worst_distance[i] = worst_distance[i]**0.5
    best_distance[i] = best_distance[i]**0.5

#calculate similarity
worst_similarity = np.zeros(8)
best_similarity = np.zeros(8)
for i in range(norm_weigh_df.shape[0]):
    worst_similarity[i] = worst_distance[i] /(worst_distance[i]+best_distance[i])
    best_similarity[i] = best_distance[i] /(worst_distance[i]+best_distance[i])

# output the ranking 
def ranking(data):
    return [i+1 for i in data.argsort()]

ranking(worst_similarity)
ranking(best_similarity)


# visulization 
from matplotlib import pyplot as plt
