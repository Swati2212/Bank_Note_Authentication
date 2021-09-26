import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# Exploratory Data Analysis
df = pd.read_csv('BankNote_Authentication.csv')
df.tail()

# Checking for the missing values
df.isnull().sum()

# Indepenent and dependent features
X = df.drop('class', axis=1)
y = df['class']


# train and test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model Building
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# check accuracy
from sklearn.metrics import accuracy_score

print(f"Accuracy score---> {accuracy_score(y_test, y_pred)}")

# Creating the pickle file using serialization
pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

# Making a prediction
print(classifier.predict([[2, 3, 4, 1]]))
