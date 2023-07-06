import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,precision_score,mean_squared_error, r2_score,confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
#Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
features,target = load_wine(return_X_y=True,as_frame=True)
target = pd.DataFrame({'target':target})

#Exploratory Data Analysis
print(features)
print(target)
print(features.info())
print(target.info())
print(f'The total missing values in the features = \n{features.isna().sum()}')

#Plot the Countplot of the classes in the target variable
ax = sns.countplot(x='target',order=target.target.value_counts(ascending=False).index ,data=target)
abs_values = target.target.value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0],labels=abs_values)
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Create the model
model = LogisticRegression()
# Train the model using the training data
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print('Classification Report:')
print(cr)

#Plot the heatmap
plt.figure(figsize=(15,15))
sns.heatmap(features.corr(),cmap="YlGnBu", annot=True)
sns.despine()
plt.show()

#Hyperparameterization
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 500]
}
# Create GridSearchCV object
grid_search = GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1,cv=5)

# Fit the grid search model
grid_search.fit(features, target)

# Print the best parameters and best score
print(f"Best estimator = {grid_search.best_estimator_}")
print(f"Best Params = {grid_search.best_params_}" )
print(f"Best Score = {grid_search.best_score_}")