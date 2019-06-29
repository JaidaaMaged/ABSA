import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
#read in the data using pandas
df = pd.read_csv("aspect_file.csv")
#check data has been read in properly
df.head()
#check number of rows and columns in dataset
df.shape

#create a dataframe with all training data except the target column
X = df.drop(columns=["label"])
#check that the target variable has been removed
print(X.head())

#separate target values
y = df["label"].values
#view target values
y[0:5]

from sklearn.model_selection import train_test_split
#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5)
# Fit the classifier to the data
knn.fit(X_train,y_train)

#show first 5 model predictions on the test data
print(knn.predict(X_test)[0:5])
predictions = knn.predict(X_test)
#check accuracy of our model on the test data
print(knn.score(X_test, y_test))

from sklearn.model_selection import cross_val_score
import numpy as np
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=5)
#train model with cv of 5
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print("‘cv_scores mean:{}’".format(np.mean(cv_scores)))
pos =0
neg = 0
neu =0
for i in predictions:
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

print("  O:  " + y_test + "  P: "+ predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.pie(statistics, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax.set_title('Product Reviews Summary')


plt.show()

