import pandas as pd
import aiohttp
import asyncio
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Загрузка тестовых данных test.csv
data_test = pd.read_csv('test.csv')
# Преобразование категориальных признаков в числовые
Vehicle_Age_category = ['< 1 Year', '1-2 Year', '> 2 Years']
Vehicle_Age_number = list(range(len(Vehicle_Age_category)))
for idx, column_name in enumerate(Vehicle_Age_category):
    data_test['Vehicle_Age'] = data_test['Vehicle_Age'].apply(lambda x: Vehicle_Age_number[idx] if x == column_name else x)

data_test['Gender'] = data_test['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
data_test.rename(columns={'Gender': 'Is_Male'}, inplace=True)
data_test['Vehicle_Damage'] = data_test['Vehicle_Damage'].apply(lambda x: 1 if x == 'Yes' else 0)

# Преобразование типов данных
int_columns = ['Region_Code', 'Policy_Sales_Channel']
for col_name in int_columns:
    data_test[col_name] = data_test[col_name].astype('int64')

# Удаление бесполезных признаков
columns_to_remove = ['id', 'Region_Code', 'Policy_Sales_Channel', 'Driving_License', 'Annual_Premium', 'Vintage']
data_test = data_test.drop(columns=columns_to_remove)

# Разделение на X_test и y_test
tmp_X_test = data_test.drop(columns=['Response'])
X_test = tmp_X_test.copy()
y_test = data_test['Response']

# Загрузка scaler и модели из файлов
with open('sc_model.pkl', 'rb') as f:
    standard_scaler = pickle.load(f)

with open('clf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Масштабирование данных
X_test_scaled = pd.DataFrame(standard_scaler.transform(X_test.values), columns=X_test.columns)

# Прямое предсказание с использованием модели
y_pred_direct = pd.DataFrame(model.predict(X_test_scaled.values), columns=['Response'])

# Вывод метрик для прямого метода
print("Метрики модели ML:")
print(classification_report(y_test, y_pred_direct, digits=4))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_direct).ravel()
print('TP: {}\nTN: {}\nFP: {}\nFN: {}'.format(tp, tn, fp, fn))

# Асинхронный клиент для работы с API
async def predict_batch_api(session, api_url, batch):
    try:
        async with session.post(api_url, json=batch) as response:
            if response.status == 200:
                result = await response.json()
                return result["predictions"]
            else:
                print(f"API request failed with status code {response.status}")
                return None
    except Exception as e:
        print(f"Connection error: {e}")
        return None

async def main():
    api_url = "http://127.0.0.1:5000/predict_model_batch" #url api /predict_model_batch запущненой локально
    batch_size = 1000
    y_pred_api = []

    async with aiohttp.ClientSession() as session:
        # Разделение данных на батчи
        batches = [X_test_scaled.iloc[i:i + batch_size].to_dict(orient='records') for i in range(0, len(X_test_scaled), batch_size)]

        # Обработка батчей
        for batch in tqdm(batches, desc="Processing batches"):
            predictions = await predict_batch_api(session, api_url, batch)
            if predictions:
                y_pred_api.extend(predictions)

    return y_pred_api

# Запуск асинхронного кода
y_pred_api = asyncio.run(main())

# Преобразование предсказаний из строкового формата в числовой
y_pred_api_numeric = pd.DataFrame(
    [1 if pred == "Response negative" else 0 for pred in y_pred_api],
    columns=['Response']
)

# Вывод метрик для api
print("\nМетрики API:")
print(classification_report(y_test[:len(y_pred_api_numeric)], y_pred_api_numeric, digits=4))

tn, fp, fn, tp = confusion_matrix(y_test[:len(y_pred_api_numeric)], y_pred_api_numeric).ravel()
print('TP: {}\nTN: {}\nFP: {}\nFN: {}'.format(tp, tn, fp, fn))

# Сравнение предсказаний
mismatched_predictions = y_pred_direct.iloc[:len(y_pred_api_numeric)][y_pred_direct.iloc[:len(y_pred_api_numeric)]['Response'] != y_pred_api_numeric['Response']]
print(f"\nКоличество несовпадений: {len(mismatched_predictions)}")
if len(mismatched_predictions) > 0:
    print("Количество несовпадений:")
    print(mismatched_predictions)