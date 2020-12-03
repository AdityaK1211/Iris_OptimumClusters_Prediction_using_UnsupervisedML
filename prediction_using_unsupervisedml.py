# -*- coding: utf-8 -*-
"""Prediction_using_UnsupervisedML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_R8Uq0Nj7gyQErrZZfVeDM-KMfIaq3k5

# Task 2 - Prediction using Unsupervised ML
## (Level - Beginner)

Author : Aditya K. Kataria
Data Science & Business Analytics Internship
GRIP December2020

Aim: From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually. <br>
Dataset: Data can be found at https://bit.ly/3kXTdox <br>

"""

# Importing all the important Libraries
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates, andrews_curves
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# %matplotlib inline

# Loading Dataset
df = pd.read_csv('Iris.csv')
print('Shape:', df.shape)
print(df.head())

# Visualizing Data
# Pie chart
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
sizes = [(df['Species'] == 'Iris-setosa').sum(),
         (df['Species'] == 'Iris-versicolor').sum(),
         (df['Species'] == 'Iris-virginica').sum()]
# colors
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
fig1, ax1 = plt.subplots()
patches, texts, autotexts = ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
for text in texts:
    text.set_color('black')
for autotext in autotexts:
    autotext.set_color('black')
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.savefig('plots/piechart.png')
plt.show()

# Andrews Curves
plt.figure(figsize=(10, 5))
andrews_curves(df.drop("Id", axis=1), "Species")
plt.title('Andrews Curves Plot', fontsize=5, fontweight='bold')
plt.xlabel('Features', fontsize=5)
plt.ylabel('Features values', fontsize=10)
plt.legend(loc=1, prop={'size': 10}, frameon=True, shadow=True, facecolor="white", edgecolor="black")
plt.savefig('plots/andrewscurves.png')
plt.show()

# Parallel Coordinates
plt.figure(figsize=(10, 5))
parallel_coordinates(df.drop("Id", axis=1), "Species")
plt.title('Parallel Coordinates Plot', fontsize=5, fontweight='bold')
plt.xlabel('Features', fontsize=5)
plt.ylabel('Features values', fontsize=10)
plt.legend(loc=1, prop={'size': 10}, frameon=True, shadow=True, facecolor="white", edgecolor="black")
plt.savefig('plots/parallelcoordinates.png')
plt.show()

# Pairplot
plt.figure()
sns.pairplot(df.drop("Id", axis=1), hue="Species", height=3, markers=["o", "s", "D"])
plt.title('Pairplot', fontsize=5, fontweight='bold')
plt.savefig('plots/pairplot.png')

# Pairgrid
plt.figure()
sns.pairplot(df.drop("Id", axis=1), hue="Species", height=7, diag_kind="kde")
plt.title('Pairplot', fontsize=5, fontweight='bold')
plt.savefig('plots/pairgrid.png')

# Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title('Heatmap', fontsize=5, fontweight='bold')
plt.savefig('plots/heatmap.png')
plt.show()

"""## Preparing the data
Here, we store the values of attributes 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm' and 'PetalWidthCm' in X and label 'Species' in y.
"""
# Dividing the dataset columns. 
# Attributes in X and Labels in y
X = df.iloc[:, [1, 2, 3, 4]].values
y = df['Species'].values

# Label Encoder
# Iris-setosa correspond to 0, Irisversicolor correspond to 1 and Iris-virginica correspond to 2.
encoder = LabelEncoder()
y_label = encoder.fit_transform(y)

# Elbow Method
# Finding optimal K value
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}

for i in range(1, 10):
    # Building and fitting the model
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(X)
    kmeans.fit(X)

    distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeans.inertia_)

    mapping1[i] = sum(np.min(cdist(X, kmeans.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[i] = kmeans.inertia_

# Distortions
for key, val in mapping1.items():
    print(str(key) + ' : ' + str(val))

plt.plot(range(1, 10), distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.savefig('distortions.png')
plt.show()

# Inertia
for key, val in mapping2.items():
    print(str(key) + ' : ' + str(val))

plt.plot(range(1, 10), inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.savefig('inertia.png')
plt.show()

# K-Means Clustering
# Predicting labels
kmeans = KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(X)
print(y_pred)

# Calculating cluster centers
centers = kmeans.cluster_centers_
print(centers)

# Comparison of Actual and K-Means Predicted Labels
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=y_label)
axes[1].scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=y_pred, cmap=plt.cm.Set1)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('K_Means', fontsize=18)
plt.savefig('actualvspredicted.png')
plt.show()

# Visualising the clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=100, c='red', label='Iris-setosa')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s=100, c='blue', label='Iris-versicolour')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s=100, c='green', label='Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')
plt.savefig('clusters.png')
plt.legend()

# Evaluation
print('Classification Report\n', classification_report(y_label, y_pred))
accuracy = accuracy_score(y_label, y_pred) * 100
print('K-Meprans Accuracy:', str(round(accuracy, 2)) + '%')

# Confusion Matrix
cm = confusion_matrix(y_label, y_pred)
print(cm)

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index=['setosa', 'versicolor', 'virginica'],
                     columns=['setosa', 'versicolor', 'virginica'])
sns.heatmap(cm_df, annot=True)
plt.title('Accuracy:{0:.4f}'.format(accuracy_score(y_label, y_pred)))
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('confusionMatrix.png')
plt.show()
