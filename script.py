import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
print(transactions.isFraud.value_counts())
print("There are 282 fraudulent transactions.")

# Summary statistics on amount column
print(transactions.amount.describe())

plt.hist(transactions.amount)
plt.title("Distribution of Amounts Charged")
plt.show()
plt.clf()
print("The distribution is very right-skewed as there are several very large transactions.")

# Create isPayment field
transactions['isPayment']= 0
transactions['isPayment'][transactions['type'].isin(['PAYMENT', 'DEBIT'])]= 1

# Create isMovement field
transactions['isMovement']= 0
transactions['isMovement'][transactions['type'].isin(['CASH_OUT', 'TRANSFER'])]

# Create accountDiff field
transactions['accountDiff']= np.abs(transactions.oldbalanceOrg - transactions.oldbalanceDest)

# Create features and label variables
# label column in this dataset is the isFraud field
features= transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label= transactions[['isFraud']]

# Split dataset
x_train, x_test, y_train, y_test= train_test_split(features, label, test_size=0.3)

# Normalize the features variables
scaler= StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

# Fit the model to the training data
model= LogisticRegression()
model.fit(x_train, y_train)

# Score the model on the training data
print(model.score(x_train, y_train))

# Score the model on the test data
print(model.score(x_test, y_test))

# Print the model coefficients
print(model.coef_)
print('Features are amount, isPayment, isMovement, and accountDiff. Most important is amount, least important is isMovement.')

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction= np.array([1270.0, 1.0, 1.0, 125000.0])

# Combine new transactions into a single array
sample_transactions= np.array([transaction1, transaction2, transaction3, your_transaction])

# Normalize the new transactions
sample_transactions= scaler.transform(sample_transactions)

# Predict fraud on the new transactions
print(model.predict(sample_transactions))

# Show probabilities on the new transactions
print(model.predict_proba(sample_transactions))
