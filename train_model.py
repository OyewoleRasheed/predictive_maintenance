import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#import matplotlib.pyplot as plt
import joblib

#load dataset
data = pd.read_csv(r'data\ai4i2020.csv')
# print(data.head())
# print(data.shape)
# print(data.info())
#print(data['Machine failure'].value_counts())

# le = LabelEncoder()
# data['Type'] = le.fit_transform(data['Type'])
# data['Product ID'] = le.fit_transform(data['Product ID'])

X= data.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF','Type', 'UDI','Product ID'], axis=1)
y = data['Machine failure']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight = 'balanced', random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)


# plt.figure(figsize=(10,6))
# plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
# plt.xlabel('Importance')
# plt.title('Feature Importance for Machine Failure Prediction')
# plt.gca().invert_yaxis()  # Highest importance on top
# plt.show()

# joblib.dump(model, 'machine_failure_model.pkl')

