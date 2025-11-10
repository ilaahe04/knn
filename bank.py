# Lazımi kitabxanaların yüklənməsi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

# Datasetin oxunması və ilkin baxış
df=pd.read_csv('german_credit_risk.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.info()
df.isnull().sum()

# Boş dəyərlərin doldurulması
df['Saving accounts'].value_counts()
df['Saving accounts'].fillna(value=df['Saving accounts'].mode()[0], inplace=True)

df['Checking account'].value_counts()
df['Checking account'].fillna(value=df['Checking account'].mode()[0], inplace=True)

# Boşluqların qalıb-qalmadığını yenidən yoxlanılması
df.isnull().sum()

# İlkin statistik analiz və vizuallaşdırma
df.describe()
df.columns

# Histogram: Yaş paylanması
sns.histplot(df['Age'], kde=True);

# Scatterplot: Kreditin müddəti və məbləği arasındakı əlaqə
sns.scatterplot(data=df, x='Duration', y='Credit amount');

# Barplot: Risk və Cinsə görə orta kredit məbləği
sns.barplot(data=df, x='Risk', y='Credit amount', estimator='mean', ci=None, hue='Sex');

# Kateqorik dəyişənlərin kodlaşdırılması
df.head()
# Cins sütununu rəqəmlərə çeviririk
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
# One-hot encoding ilə kodlaşdırma
df.Housing.value_counts()
df = pd.get_dummies(df, columns=['Housing'], drop_first=False, dtype='int')
df['Saving accounts'].value_counts()
df = pd.get_dummies(df, columns=['Saving accounts'], drop_first=False, dtype='int')
df['Checking account'].value_counts()
df = pd.get_dummies(df, columns=['Checking account'], drop_first=False, dtype='int')

# Purpose sütununu LabelEncoder ilə ədədi formata çeviririk
encoder=LabelEncoder()
df['Purpose'] = encoder.fit_transform(df['Purpose'])

# Risk sütununu binary formata salırıq
df.Risk.value_counts()
df['Risk'] = df['Risk'].map({'bad': 0, 'good': 1})

# Model üçün verilənlərin hazırlanması
X=df.drop('Risk', axis=1)
y=df['Risk']

# Train və test hissələrinə bölmə
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=42,test_size=20)

# Miqyaslaşdırma (StandardScaler)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# KNN modelinin qurulması və qiymətləndirilməsi
# K=3 üçün model
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Nəticələri qiymətləndir
cr=classification_report(y_test, y_pred)
print('Classification report')
print(cr)

# K=5 üçün model
knn2=KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train, y_train)
y_pred = knn2.predict(X_test)
cr=classification_report(y_test, y_pred)
print('Classification report')
print(cr)

# K=7 üçün model
knn3=KNeighborsClassifier(n_neighbors=7)
knn3.fit(X_train, y_train)
y_pred = knn3.predict(X_test)
cr=classification_report(y_test, y_pred)
print('Classification report')
print(cr)
