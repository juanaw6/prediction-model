from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd

# Load the dataset
data = pd.read_csv('test.csv')

# Separate features and target
X = data.drop(columns=['target_class'])
y = data['target_class']

# Apply OneHotEncoder to categorical features
encoder = OneHotEncoder()
label_encoder = LabelEncoder()
X_encoded = encoder.fit_transform(X).toarray()
y_encoded = label_encoder.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Define models to evaluate
models = {
    # 'Random Forest': RandomForestClassifier(random_state=42),
    # 'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    # 'AdaBoost': AdaBoostClassifier(random_state=42),
    # 'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    # 'Decision Tree': DecisionTreeClassifier(random_state=42),
    # 'Extra Trees': ExtraTreesClassifier(random_state=42),
    # 'HistGradient Boosting': HistGradientBoostingClassifier(random_state=42),
    # 'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    # 'LightGBM': LGBMClassifier(random_state=42),
    # 'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
    'MLP Classifier': MLPClassifier(random_state=42, max_iter=1000)
}

# Evaluate each model
for name, model in models.items():
    print(f"Evaluating {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(f"{name} Report:\n{report}\n")