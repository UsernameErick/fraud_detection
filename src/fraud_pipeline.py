import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "../data", "fraudTest.csv")
df = pd.read_csv(dataset_path)

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time']) # конвертируем в datetime objects
df['hour'] = df['trans_date_trans_time'].dt.hour # достаем только час

df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 4)).astype(int) # все что между 22 и 4 утра будет 1, иначе 0
df['dob'] = pd.to_datetime(df['dob']) # dob - date of birth
df['age'] = (df['trans_date_trans_time'].dt.year - df['dob'].dt.year) # находим возраст людей вычитая дату рождения от настоящего времени(времени транзакции)

# добавим признак merchant
merchant_stats = df.groupby('merchant')['is_fraud'].agg(['count', 'sum', 'mean']).sort_values('mean', ascending=False)
print("merchants stats:\n", merchant_stats.head(20))

features = ['amt', 'hour', 'is_night', 'age', 'category', 'gender', 'state']

X = df[features]
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_features = ['amt', 'hour', 'is_night', 'age']
categorical_features = ['category', 'gender', 'state']

param_grid = {'model__max_depth': [3, 5, 7], 'model__n_estimators': [100, 300, 500]} # готовим сетку параметров для модели через gridsearch

# grid search cv
scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])
preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough') 
pipeline = Pipeline([('preprocessor', preprocessor), ('model', XGBClassifier(random_state=42, scale_pos_weight = scale_pos_weight, eval_metric = 'logloss'))])
grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1, verbose=2) # cv - кол-во фолдов, n_jobs - задейств. ядра, verbose - показ. прогресс
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)

# pred = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:, 1]
pred = (proba >= 0.99).astype(int) # добавим трешхолд 0.99
print(f"classification_report with max_depth = {7}, n_est = {500}:\n", classification_report(y_test, pred))
# print("ROC-AUC:", roc_auc_score(y_test, proba))
# print("PR-AUC:", average_precision_score(y_test, proba))

# pipeline настроен
pipeline_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
feature_importance_df = pd.DataFrame({'feature': pipeline_feature_names, 'importance': pipeline.named_steps['model'].feature_importances_}).sort_values(by='importance', ascending=False)
# print(feature_importance_df.head(10))

# print(pipeline.named_steps['preprocessor'].transform(X_train).shape)
# print(pipeline.named_steps['model'].get_params())
# print(pipeline.get_params()['model__max_depth'])
