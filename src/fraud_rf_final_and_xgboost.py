import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "../data", "fraudTest.csv")
df = pd.read_csv(dataset_path)

# EDA

fraud_by_category = (df.groupby('category')['is_fraud'].mean().sort_values(ascending=False))

# time analysis (узнать, в какие часы чаще всего происходило мошенничество)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time']) # конвертируем в datetime objects
df['hour'] = df['trans_date_trans_time'].dt.hour # достаем только час
fraud_by_hour = (df.groupby('hour')['is_fraud'].mean().sort_values(ascending=False))

# сделаем новую фичу - ночь
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 4)).astype(int) # все что между 22 и 4 утра будет 1, иначе 0
# age dependency
df['dob'] = pd.to_datetime(df['dob']) # dob - date of birth
df['age'] = (df['trans_date_trans_time'].dt.year - df['dob'].dt.year) # находим возраст людей вычитая дату рождения от настоящего времени(времени транзакции)

# ML
features = ['amt', 'hour', 'is_night', 'age']
target = 'is_fraud'

X = df[features]
y = df[target]

# создадим dummies от category. dummies переводит все категории в векторы размерностью как кол-во элементов в категории(как в tf-idf). каждый вектор выглядит типа 0 0 0 1 0 и является значением category_(имя той категории на которой стоит 1 а не 0, например category_grocery_pos)
category_dummies = pd.get_dummies(df['category'], prefix="category")

# соберем новый X для более качественного обучения. теперь там будут и данные категорий в виде dummies, а значит больше признаков. (ну и признаки которые мы создали раннее)
X = pd.concat([df[['amt', 'hour', 'is_night', 'age']], category_dummies], axis=1)

upd_fraud_rate = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False) # еще раз. .mean() ищет AVG во fraud, например [0, 0, 1, 0] -> 0+0+1+0 / 4 = 0.25 - fraud rate.

# создадим дамми-таблицы
gender_dummies = pd.get_dummies(df['gender'], prefix="gender")
state_dummies = pd.get_dummies(df['state'], prefix='state')

# соберем обновленный X с новыми дамми
X = pd.concat([df[['amt', 'hour', 'is_night', 'age']], category_dummies, gender_dummies, state_dummies], axis = 1)

# обучим снова RandomForest, надеясь на еще лучший результат
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
print("Updated RF classification_report 2:\n", classification_report(y_test, rf_predictions))
print(confusion_matrix(y_test, rf_predictions))
# результаты стали еще лучше по всем параметрам. модель и нашла больше и ошибаться стала меньше. ложных тревог меньше в 2 раза

# roc-auc
fraud_proba = rf.predict_proba(X_test)[:,1]

roc = roc_auc_score(y_test, fraud_proba)
print("ROC-AUC probas:\n", roc)

# pr-auc
pr_auc = average_precision_score(y_test, fraud_proba)
print("PR-AUC probas:\n", pr_auc)

# ROC-AUC probas: 0.9813688470654246
# PR-AUC probas: 0.8856311070023025
# 1       0.94      0.74      0.83 RF

importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
#print("Updated feature importances 2:\n", importance.sort_values('importance', ascending=False).head(20))

# посмотрим на самые мошеннические штаты
most_fraud_states = df.groupby('state')['is_fraud'].mean().sort_values(ascending=False)
state_stats = df.groupby('state').agg({'is_fraud': ['count', 'sum', 'mean']})

# попробуем Gradient Boost, будет лучше или нет
# gb = GradientBoostingClassifier(random_state=42)

# gb.fit(X_train, y_train)
# gb_pred = gb.predict(X_test)
# print("GradientBoosting classification_report:\n", classification_report(y_test, gb_pred))

# gb_proba = gb.predict_proba(X_test)[:, 1]
# print("ROC-AUC:", roc_auc_score(y_test, gb_proba))
# print("PR-AUC:", average_precision_score(y_test, gb_proba))

# ROC-AUC: 0.9745806649842635
# PR-AUC: 0.6328037091565938
# 1       0.81      0.70      0.75 gradientboost оказался слабее

# попробуем XGBoost
scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1]) # поделить то что чаще повторяется на то что меньше - кол-во честных / кол-во мошенников

xgb = XGBClassifier(random_state = 42, scale_pos_weight = scale_pos_weight, eval_metric = 'logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print("XGBoost classification_report\n", classification_report(y_test, xgb_pred))

xgb_proba = xgb.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, xgb_proba))
print("PR-AUC:", average_precision_score(y_test, xgb_proba))

# ROC-AUC: 0.9954477992433003
# PR-AUC: 0.9111333563226529
# 1       0.36      0.95      0.52 XGBoost

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]:
    y_pred_thresholds = (xgb_proba > threshold).astype(int)
    print(f"THRESHOLD {threshold}:")
    print(classification_report(y_test, y_pred_thresholds))
# ^
# THRESHOLD 0.95:
#           1       0.79      0.86      0.83 XGBoost
# THRESHOLD 0.99:
#           1       0.96      0.79      0.86 лучше по всем параметрам чем прошлый порог 0.95 и RandomForest

xgb_feature_importances = pd.DataFrame({'feature': X.columns, 'importance': xgb.feature_importances_})
print("xgb feature importances\n", xgb_feature_importances.sort_values(by='importance', ascending=False).head(10))