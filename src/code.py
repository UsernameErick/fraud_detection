import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "../data", "fraudTest.csv")
df = pd.read_csv(dataset_path)

#print(df.head())

# EDA
print(df.groupby('is_fraud')['amt'].describe())

print("Fraud distribution:\n", df['is_fraud'].value_counts(normalize=True)) # 0.3% fraud

print("Is every record not null?\n", df.isnull().sum())

print("Fraud mean amount\n", df.groupby('is_fraud')['amt'].mean())

plt.hist(df[df['is_fraud'] == 0]['amt'], bins=50, alpha = 0.5, label='Normal')
plt.hist(df[df['is_fraud'] == 1]['amt'], bins=50, alpha = 0.5, label='Fraud')
plt.xlim(0, 1000) # ограничения для X-оси
plt.ylabel("amount")
plt.xlabel("transactions value")
plt.legend()
plt.show()
# здесь мы видим, что подавляющее кол-во мошеннических переводов является крупным переводом

fraud_by_category = (df.groupby('category')['is_fraud'].mean().sort_values(ascending=False))

print("The most frequent categories with fraud:\n", fraud_by_category.head(10))
# можно сделать вывод, что большинство мошенничеств приходятся на шоппинг, продукты, как онлайн, так и оффлайн. можно сделать небольшой вывод, что шопиться оффлайн более рискованно.

# time analysis (узнать, в какие часы чаще всего происходило мошенничество)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time']) # конвертируем в datetime objects
df['hour'] = df['trans_date_trans_time'].dt.hour # достаем только час
fraud_by_hour = (df.groupby('hour')['is_fraud'].mean().sort_values(ascending=False))
print("Fraud rate by hours:\n", fraud_by_hour.head(10))
# делаем хороший вывод: большинство преступлений было сделано в вечернее-ночное время(22, 23, 3, 0, 2, 1), а потом идет значительный спад со всеми остальными часами в сутках. ночью люди менее внимательны, позже замечают ошибку
# визуализация
fraud_by_hour.plot(kind='bar', figsize=(10, 6))
plt.title("Fraud rate by Hour")
plt.show()

# сделаем новую фичу - ночь
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 4)).astype(int) # все что между 22 и 4 утра будет 1, иначе 0
print("Fraud rate by Night-time:\n", df.groupby('is_night')['is_fraud'].mean())

# age dependency
df['dob'] = pd.to_datetime(df['dob']) # dob - date of birth
df['age'] = (df['trans_date_trans_time'].dt.year - df['dob'].dt.year) # находим возраст людей вычитая дату рождения от настоящего времени(времени транзакции)
print("Mean frauded age:\n", df.groupby('is_fraud')['age'].mean())
# мы делаем вывод, что возраст не сильно влияет на то, насколько успешно вас одурачат. 46.6 <-> 48.7

# ML
features = ['amt', 'hour', 'is_night', 'age']
target = 'is_fraud'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # нельзя fit ибо начнет обучаться на тестовых данных

model = LogisticRegression(class_weight='balanced') # class weight balanced берется для того чтобы модель не взяла и не сказала что мошенников вообще нет(их меньше 1%, ей так выгоднее сказать)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

print("LR classification_report:\n", classification_report(y_test, predictions))

# confusion matrix
cm = confusion_matrix(y_test, predictions)
print(cm)
# confusion matrix показала как и в classification report у модели очень агрессивный precision для fraud(0.02 - он называет мошенниками почти всех и часто ошибается), 21957 раз он назвал обычных мошенниками
#   true    0 |   |   |
#           1 |   |   |
#               0   1
#             prediction

# попытаемся сравнить результат с моделью RandomForest
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
print("RF classification_report:\n", classification_report(y_test, rf_predictions))
cm_rf = confusion_matrix(y_test, rf_predictions)
print(cm_rf)
# результат улучшился, precision повысился ценой recall. благодаря этому модель чаще угадывает правильно, а значит меньше срабатывает "ложный вызов" и меньше страдают клиенты, что для банка важнее

# predict-proba - не 0/1, а вероятность fraud
fraud_probs = rf.predict_proba(X_test)[:,1] # если без [:, 1] то получим массив вероятностей [[0.3, 0.7], [0.99, 0.01], ...] не фрод/фрод

custom_predictions = (fraud_probs > 0.9).astype(int) # порог подозрения ниже, менее осторожная
print("prediction-probability:\n", classification_report(y_test, custom_predictions))
cm_rf_proba = confusion_matrix(y_test, custom_predictions)
print(cm_rf)
# threshold 0.3 precision 0.57 recall 0.57 f1 0.57
# threshold 0.8 precision 0.93 recall 0.34 f1 0.50
# threshold 0.9 precision 0.97 recall 0.27 f1 0.42
# чем выше осторожность, тем выше precision и ниже recall. находит мало(recall), но если находит то вероятность что он прав высока(precision)
# слишком высокий precision тоже не очень хорошо, т.к. страдает recall. поэтому делают уровни риска. 
# fraud probability > 0.95 - block, f prob 0.7-0.95 - manual review, f prob 0.5-0.7 - sms verification

# создадим risk levels. low, medium, high
risk_level = []
for prob in fraud_probs:
    if prob < 0.3:
        risk_level.append('low')
    elif prob < 0.7:
        risk_level.append('medium')
    else:
        risk_level.append('high')

pd.Series(risk_level).value_counts() # превратить список в множество, а value_counts выведет кол-во low, medium и high

results = pd.DataFrame({'risk_level': risk_level, 'actual_fraud': y_test.values}) # создадим датафрейм с вероятностями от предикта fraud и реальными ответами из y_test(который тоже series)
fraud_rate = (results.groupby('risk_level')['actual_fraud'].mean())
print(fraud_rate)
# из всех операций, что модель пометила как low risk - 0.1% реально fraud. medium - 32%, high risk - 84% операций оказались fraud

# выведем топ рискованных категорий
high_risk = results.copy()
high_risk['category'] = (X_test.index.map(df['category'])) # берем индексы у X_test, по ним находим соответствующие category, и именно те категории с индексами от признаков X_test мы заталкиваем в high_risk
print("Top risky categories:\n", high_risk[high_risk['risk_level'] == 'high']['category'].value_counts().head(10))
# вывод: шопинг и другие покупки в интернете - самые рискованные. везде, где сложно найти продавца товара/услуги либо официальный филиал, шанс встретить мошенника выше 

# feature importance
importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}) # X = df[features]. просто создана таблица важности признаков
importance = importance.sort_values(by='importance', ascending=False)
print("Feature Importance:\n", importance)
# самые важные признаки при выявлении мошенничества по версии RandomForest - сумма потраченных денег(подавляющая важность), время суток+часы во время операции, возраст жертвы.

# дальнейшие строки будут нацелены на улучшение распознавания, поиск новых признаков, использование новых моделей

# создадим dummies от category. dummies переводит все категории в векторы размерностью как кол-во элементов в категории(как в tf-idf). каждый вектор выглядит типа 0 0 0 1 0 и является значением category_(имя той категории на которой стоит 1 а не 0, например category_grocery_pos)
category_dummies = pd.get_dummies(df['category'], prefix="category")
print(category_dummies.head())
print(category_dummies.shape)
# ^ shape 555 719 строк (сколько и транзакций в датафрейме) и 14 категорий магазинов

# соберем новый X для более качественного обучения. теперь там будут и данные категорий в виде dummies, а значит больше признаков. (ну и признаки которые мы создали раннее)
X = pd.concat([df[['amt', 'hour', 'is_night', 'age']], category_dummies], axis=1)

# теперь с новым X обучим и протестируем RandomForest и посмотрим на обновленный результат!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
print("Updated RF classification report:\n", classification_report(y_test, rf_predictions))
print(confusion_matrix(y_test, rf_predictions))
# обновленный результат показал очень большой прорыв в ориентировании модели. она ловит гораздо больше мошенников(+109) но и по точности предсказания не отстает. precision 0.88 recall 0.73 f1 0.80
# хорошие признаки очень важны. модель смогла найти больше закономерностей и больше мест за которые можно зацепиться

importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
importance = importance.sort_values(by='importance', ascending=False).head(20)
print("Updated feature importance:\n", importance)
# старые фичи все еще подавляющие по значимости для модели RandomForest

# сейчас я хочу выяснить какие категории имеют наибольший fraud rate среди друг друга
upd_fraud_rate = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False) # еще раз. .mean() ищет AVG во fraud, например [0, 0, 1, 0] -> 0+0+1+0 / 4 = 0.25 - fraud rate.
print("Updated fraud rate:\n", upd_fraud_rate)
# shopping_net имеет как высокую значимость для модели, так и высокую частоту фрода(1.21%). однако gas_transport имеет высокую важность для модели, но относительно низкую
# частоту фрода(0.2%). это наталкивает сделать вывод, что модель смотрит на комбинацию признаков, а не по отдельности на каждый.

# дальше я хочу развить остальные признаки и попытаться разобраться с ними
print(df['gender'].value_counts())
print(df['state'].nunique())
print(df['job'].nunique())
print(df['merchant'].nunique())

# создадим дамми-таблицы
gender_dummies = pd.get_dummies(df['gender'], prefix="gender")
state_dummies = pd.get_dummies(df['state'], prefix='state')
print(gender_dummies.shape)
print(state_dummies.shape)

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

importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
print("Updated feature importances 2:\n", importance.sort_values('importance', ascending=False).head(20))

# посмотрим на самые мошеннические штаты
most_fraud_states = df.groupby('state')['is_fraud'].mean().sort_values(ascending=False)
print("Most fraud states:\n", most_fraud_states.head(10))
# Аляска, Коннектикут, Айдахо, Гавайи, Монтана, Вашингтон. первые два вырываются вперед 1.6% и 1.2%, третье место 0.8%. рано делать выводы, нужно смотреть распределение fraud по всем штатам
state_stats = df.groupby('state').agg({'is_fraud': ['count', 'sum', 'mean']})
print("state_stats:\n", state_stats.sort_values(('is_fraud', 'mean'), ascending=False).head(10))
# видна неровность в кол-ве штатов на кол-во мошенничеств. на Аляске мало транзакций в принципе, поэтому любой fraud может сильно перекосить процент mean(avg) fraud'а. 
# по-хорошему надо ставить фильтр state_stats[state_stats[('is_fraud', 'count')] > 5000], чтобы в анализ попадались штаты где достаточно данных для здравой оценки
print(X.shape)