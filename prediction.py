import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns 

data=pd.read_csv("dataset.csv")

print(data.head(5))

print(data.columns)

print(data.shape)

print(data.info())

#print(data.isnull().sum())


#print(data.dtypes)


data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors='coerce')

data=data.drop('customerID',axis=1)

data=data.dropna()

print(data.dtypes)
print(data.isnull().sum())


print(data['Churn'].value_counts())


sns.countplot(x='Churn',data=data)

plt.title('Churn Count')
plt.show()

#univariate plot
 
sns.histplot(data['MonthlyCharges'],kde=True,color='blue')
plt.title("Distribution of Monthly charges ")
plt.show()


sns.histplot(data['tenure'],kde=True,color='green')
plt.title("Distribution of Tenure")
plt.show()

#bivariate plot

sns.boxplot(x='Churn',y='MonthlyCharges',data=data)
plt.title("Monthlycharges vs Churn")
plt.show()


sns.boxplot(x='Churn',y='tenure',data=data)
plt.title("tenure vs Churn")
plt.show()

#categorieal plot

sns.countplot(x='InternetService',hue='Churn',data=data)
plt.title("InternetService vs Churn")
plt.show()

sns.countplot(x='Contract',hue='Churn',data=data)
plt.title("Contract vs Churn")
plt.show()

sns.countplot(x='PaymentMethod',hue='Churn',data=data)
plt.title("PaymentMethod vs Churn")
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le=LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] =le.fit_transform(data[column])

print(data.head(5))

x=data.drop('Churn',axis=1)
y=data['Churn']



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x.shape)
print(y.shape)


# Step 7: Model Building

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

import pickle

# Save the trained model
pickle.dump(model, open('model.pkl', 'wb'))
