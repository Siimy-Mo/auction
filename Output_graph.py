import os
import matplotlib.pyplot as plt

path = "./Performance_epoch.txt"
Lines = ['HR','MRR','NDCG']

textPath = path

with open(textPath,'r') as f:
  dataset=[[]for j in range(3)]
  epoch=[]
  epo=0
  for line in f.readlines():
    epoch.append(epo)
    epo+=1

    data=line.split(': ')
    
    for i in range(3):
      dataset[i].append(float(data[i+2][:8]))
    print(dataset)
  
  plt.title('Histogram of Criterias')
  plt.xlabel('Epoch')
  plt.ylabel('Value')
  # print(len(epoch), len(dataset[0]))
  for i in range(3):
    plt.plot(epoch,dataset[i],marker='o')
  plt.legend(Lines)
  plt.show()