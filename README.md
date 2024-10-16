# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import Libraries: Load required libraries (NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn).
2.Load Dataset: Use load_iris() to obtain the Iris dataset.
3.Prepare Data: Create a DataFrame and split it into features (X) and target (y); then split into training and testing sets.
4.Initialize Classifier: Create an instance of SGDClassifier.
5.Train Model: Fit the classifier on training data.
6.Make Predictions: Predict the species using test data.
7.Evaluate: Calculate accuracy and display the confusion matrix.
8.Output Results: Print accuracy and visualize the confusion matrix.
```

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: 
RegisterNumber:  
*/
```

```
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier


iris = load_iris()


df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target



print(df.head())
```
## Output:
![Screenshot 2024-10-16 170626](https://github.com/user-attachments/assets/f0908f41-6895-438d-a89d-51bffe65a9a5)


```
X = df.drop('target',axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
```

```
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred = sgd_clf.predict(X_test)
sgd_clf.fit(X_train, y_train)
accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

![Screenshot 2024-10-16 170741](https://github.com/user-attachments/assets/0e9fe21a-d240-4dce-b546-e0091b62b13b)


```
cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:")
print(cm)
```

## Output

![Screenshot 2024-10-16 170843](https://github.com/user-attachments/assets/09dd0d10-6607-4589-94bf-b78bf799c86c)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
