import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
dataset = pd.read_csv("aspect_file.csv")

features = dataset.drop("label", axis=1)
targets = dataset["label"]
train_features, test_features, train_targets, test_targets = \
train_test_split(features, targets, train_size=0.6)


clf1 = SVC(kernel='linear', probability=True)
clf2 = KNeighborsClassifier(n_neighbors = 3)
clf3 = KNeighborsClassifier(n_neighbors = 5)

eclf1 = VotingClassifier(estimators=[
         ('svm', clf1), ('knn', clf2), ('neighbor', clf3)], voting='hard')
eclf1 = eclf1.fit(train_features, train_targets)
print(eclf1.predict(train_features))
np.array_equal(eclf1.named_estimators_.svm.predict(train_features),
          eclf1.named_estimators_['svm'].predict(train_features))

targets_pred = eclf1.predict(test_features)
print(eclf1.transform(train_features).shape)
print(confusion_matrix(test_targets,targets_pred))
print(classification_report(test_targets,targets_pred))

pos = 0
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
statistics_1 = []
statistics_1.append(pos)
statistics_1.append(neg)
fig, ax1 = plt.subplots()
ax1.pie(statistics_1, labels=labels, autopct='%1.1f%%')
ax1.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax1.set_title('Hard Voting')
plt.show()


eclf2 = VotingClassifier(estimators=[
    ('svm', clf1), ('knn', clf2), ('neighbor', clf3)],
       voting='soft')
eclf2 = eclf2.fit(train_features, train_targets)

print( eclf2.predict(train_features))



targets_pred = eclf2.predict(test_features)
print("  O:  " + test_targets + "  P: "+ targets_pred)
print(confusion_matrix(test_targets,targets_pred))
print(classification_report(test_targets,targets_pred))

pos = 0
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
statistics_2 = []
statistics_2.append(pos)
statistics_2.append(neg)

fig, ax2 = plt.subplots()
ax2.pie(statistics_2, labels=labels, autopct='%1.1f%%')
ax2.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax2.set_title('soft Voting')
plt.show()







eclf3 = VotingClassifier(estimators=[
   ('svm', clf1), ('knn', clf2), ('neighbor', clf3)],
       voting='soft', weights=[1,1,2],
      flatten_transform=True)
eclf3 = eclf3.fit(train_features, train_targets)
predictions = eclf3.predict(train_features)
print(eclf3.predict(train_features))
targets_pred = eclf3.predict(test_features)
print(eclf3.transform(train_features).shape)
print(confusion_matrix(test_targets,targets_pred))
print(classification_report(test_targets,targets_pred))

pos = 0
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




fig, ax3 = plt.subplots()
ax3.pie(statistics, labels=labels, autopct='%1.1f%%')
ax3.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax3.set_title('Weighted Soft Voting')

plt.show()
