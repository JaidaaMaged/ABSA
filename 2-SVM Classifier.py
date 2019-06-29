import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("aspect_file.csv")

print(dataset)



features = dataset.drop("label", axis=1)
targets = dataset["label"]
train_features, test_features, train_targets, test_targets = \
train_test_split(features, targets, train_size=0.6)

svclassifier = SVC(kernel='linear')
svclassifier.fit(train_features, train_targets)

targets_pred = svclassifier.predict(test_features)
print("  O:  " + test_targets + "  P: "+ targets_pred)
print(confusion_matrix(test_targets,targets_pred))
print(classification_report(test_targets,targets_pred))

pos =0
neg = 0
neu =0

for i in targets_pred:
    if (i == 'positive'):
        pos+=1
    elif(i=='negative'):
        neg+=1
    elif(i == 'neutral'):
        neu+=1
labels = ['positive' , 'negative']
statistics = []
statistics.append(pos)
statistics.append(neg)


import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.pie(statistics, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax.set_title('Product Reviews Summary')


plt.show()

