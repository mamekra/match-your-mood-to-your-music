import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

## the data
N = 6
roc_auc = [0.95, 0.84, 0.95, 0.72, 0.97, 0.77]
accuracy = [0.85, 0.69, 0.85, 0.74, 0.75, 0.73]
models = ["Logistic \nRegression", "Naive \nBayes", "SVM", "LSTM", "GRU", "BERT"]

## necessary variables
ind = np.arange(N)                # the x locations for the groups
width = 0.35                      # the width of the bars

## the bars
rects1 = ax.bar(ind,
                roc_auc,
                width,
                color= '#88c591',
                edgecolor='white')

rects2 = ax.bar(ind+width,
                accuracy,
                width,
                color= '#2a2747',
                edgecolor='white')

# axes and labels
ax.set_xlim(-width,len(ind)+width)
ax.set_ylim(0,1)
ax.set_ylabel('Scores')
#ax.set_title('Scores of each model')
xTickMarks = models
ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)

## add a legend
ax.legend( (rects1[0], rects2[0]), ('ROC-AUC', 'Accuracy') )

plt.show()