<!--markdown tutorial-->

MD.Redoy Sarder<br/>

---

# Titanic-Survival 

## Machine learning project

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=25&pause=1000&color=FF00FF&random=false&width=435&lines=I+am+a+Python+Developer;I+am+ML-engineer;I+am+a+Software+Developer;I+am+a+problem+solver">
<br/>

<img src="https://i0.wp.com/insights-on.com/wp-content/uploads/2021/03/10-sn56-20201221-titanicsinking-hr.jpg?fit=1440%2C804&ssl=1">

## Introduction
The sinking of the RMS Titanic in 1912 is a tragic tale of opulence and disaster. On its maiden voyage, the "unsinkable" ship struck an iceberg, leading to the loss of over 1,500 lives. Despite advanced safety features, including watertight compartments, the combination of excessive speed, insufficient lifeboats, and class-based evacuation policies proved fatal. The diverse passenger and crew makeup, ranging from wealthy elites to hopeful immigrants, adds to the poignancy. This maritime catastrophe prompted significant changes in safety regulations and remains a somber reminder of human vulnerability in the face of nature's forces, echoing through history and popular culture.

### 1 import Necessary Library

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

```
### 2 import Dataset

```
df = pd.read_csv("/kaggle/input/titanic-dataset/Titanic-Dataset.csv")
```
### 3 Data Analysis
```
df.head()
```
```
df.tail()
```
```
df.shape
```
```
df.info()
```
```
df.dtypes
```
```
df.describe()
```
```
df.ndim
```
### 4 Data cleaning and Preprocessing:
<br/>

```
# Data preprocessing
df.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'PassengerId'], axis=1, inplace=True)
df['Family'] = df['SibSp'] + df['Parch'] + 1
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
df.Embarked.replace(['S', 'C', 'Q'], [1, 2, 3], inplace=True)

```
```
df.isnull().sum()
```
```
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(1, inplace=True)
```
```
df.columns
```
```
df.head()
```
```
# Define features and target variable
x = df[['Pclass', 'Sex', 'Age', 'Embarked', 'Family']]
y = df['Survived']
```
```
df.isnull().any()
```
```
df['Survived'].value_counts()
```

<span style="color:#FFFFFF;background-color:#BB4ED8; padding:10px; width:100%"> 5| Data visualisation 📊 📉 </span>

### EDA (Exploratory Data Analysis)

```
sns.displot(df, kde=False, bins=30)
plt.show()
```
```
df.plot(kind='scatter', x='Age', y='Embarked')
plt.show()
```
```
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="Survived").map(plt.scatter, "Age", "Embarked").add_legend();
plt.show();
```
```
# sepal_length vs sepal_width boxplot

plt.figure(figsize=(15, 6))
sns.boxplot(x='Age', y='Embarked', data=df, palette='Set3')
plt.title('Age vs Embarked')
plt.xlabel('Age')
plt.ylabel('Embarked')
plt.xticks(rotation=45, ha='right')
plt.show()
```
```
# petal_length vs petal_width boxplot

plt.figure(figsize=(15, 6))
sns.boxplot(x='Age', y='Pclass', data=df, palette='Set3')
plt.title('Age vs Pclass')
plt.xlabel('Age')
plt.ylabel('Pclass')
plt.xticks(rotation=45, ha='right')
plt.show()
```
```
# sepal_length
plt.figure(figsize=(15, 8))
sns.countplot(x='Age', data=df, palette='muted')
plt.title('Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()
```
```
# sepal_width
plt.figure(figsize=(15, 8))
sns.countplot(x='Pclass', data=df, palette='muted')
plt.title('Pclass')
plt.xlabel('Pclass')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()
```
```
# petal_length
plt.figure(figsize=(15, 8))
sns.countplot(x='Sex', data=df, palette='muted')
plt.title('Sex')
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()
```
```
# sepal_width
plt.figure(figsize=(15, 8))
sns.countplot(x='Embarked', data=df, palette='muted')
plt.title('Embarked')
plt.xlabel('Embarked')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()
```
```
sns.set_style("whitegrid")
sns.pairplot(df, hue="Survived", size=3)
plt.show()
```
```
df.hist(figsize=(15,12),bins = 15)
plt.title("Features Distribution")
plt.show()
```
<span style="color:#FFFFFF;background-color:#BB4ED8; padding:10px; width:100%"> 6 | Split the Dataset </span>

```
from sklearn.model_selection import train_test_split

X = df[["Pclass", "Sex", "Age", "Embarked","Family"]]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
```
<span style="color:#FFFFFF;background-color:#BB4ED8; padding:10px; width:100%"> 7 | PCA (Principal Component Analysis) </span>

```
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_pca[0]

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X[0]
X[1]

```
# Algorithm 🔄
### (1) KNN
```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```

```
knn_classifier = KNeighborsClassifier(n_neighbors=3)
```
```
knn_classifier.fit(X_train, y_train)
```
```
train_predictions = knn_classifier.predict(X_train)

train_accuracy1 = accuracy_score(y_train, train_predictions)
```
```
test_predictions = knn_classifier.predict(X_test)

test_accuracy1 = accuracy_score(y_test, test_predictions)
```
```
print(f"Training Accuracy: {train_accuracy1}")
print(f"Testing Accuracy: {test_accuracy1}")
```
### (2) Naive Bayes classifier 🔄

```
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics
```
```
etc ... ... ...
... ... ...
... ... ...
... ... ...
```

<hr>

### Algorithm used in this data set

| Apply Algorithm List                | Email                            |
|-------------------------------------|----------------------------------|
| Linear Regression                   | syber.redoy.php.23.365@gmail.com |
| Logistic Regression                 |                                  |
| Decision Tree                       |                                  |
| Random Forest                       |                                  |
| AdaBoost (Adaptive Boosting)        |                                  |
| Gradient Boosting Machines (GBM)    |                                  |
| Support Vector Machines(SVM)        |                                  |
| K-Nearest Neighbors (KNN)           |                                  |
| Naive Bayes                         |                                  |
| Principal Component Analysis (PCA)  |                                  |

<hr>

<p align="center">
  <img width="140" src="https://user-images.githubusercontent.com/6661165/91657958-61b4fd00-eb00-11ea-9def-dc7ef5367e34.png" />
  <h2 align="center">Titanic survival</h2>
  <p align="center">🏆 Machine learning project🏆</p>
<p align="center">
  <a href="https://github.com/Redoy365?tab=repositories">
    <img src="https://img.shields.io/github/issues/ryo-ma/github-profile-trophy"/>
  </a>
  <a href="https://www.hackerrank.com/profile/syber_redoy_php">
    <img src="https://img.shields.io/github/forks/ryo-ma/github-profile-trophy"/>
  </a>
  <a href="https://redoy365.github.io/realtime/">
    <img src="https://img.shields.io/github/stars/ryo-ma/github-profile-trophy"/>
  </a>
    <a href="https://www.linkedin.com/in/md-redoy-70928b206/">
    <img src="https://img.shields.io/github/license/ryo-ma/github-profile-trophy"/>
  </a>
</p>
<p align="center">
  </a>
    <a href="https://twitter.com/FreelancerRedoy">
    <img src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fryo-ma%2Fgithub-profile-trophy"/>
  </a>
</p>
<p align="center">
  You can use this service for free. I'm looking for sponsors to help us keep up with this service❤️
</p>
<p align="center">
  <a href="https://github.com/Redoy365/ML-Project">
    <img src="https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=ff69b4"/>
  </a>
</p>

<hr>

