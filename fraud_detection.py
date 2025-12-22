import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib


data = pd.read_csv("fraud_dataset.csv")
df = pd.DataFrame(data)

print(df.head(9))

df = df.drop(['nameDest', 'nameOrig', 'newbalanceOrig', 'oldbalanceOrg', 'isFraud'], axis=1)

y = df['isFlaggedFraud']
X = df.drop(['isFlaggedFraud'], axis=1)

ct = ColumnTransformer(
    transformers=[(
        'onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        ['transaction_type', 'location']
    )],
    remainder='passthrough'
)

newX = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test,model.predict(X_test))
print(f"model accuaracy {accuracy}")

joblib.dump(model,"isFlaggedFraud_model.pkl")

joblib.dump(ct,"isFlaggedFraud_encoder.pkl")

print("model and encoder saved")

amount = float(input("Enter transaction amount: "))
transaction_type = input("Enter transaction_type (e.g., transfer, payment, withdraw, deposit): ")
location = input("Enter location (e.g., ZA, UK, CA, AU, IN): ")

predict_data = pd.DataFrame({
    'transaction_type': [transaction_type.lower()],
    'amount': [amount],
    'location': [location.upper()]
    })

predictX = ct.transform(predict_data)

prediction = model.predict(predictX)

if prediction[0] == 1:
    print("FRAUD IDENTIFIED ")
else:
    print("valid Transaction")



