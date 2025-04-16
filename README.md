# ML_1

Перечень основных файлов:
'train.csv', 'test.csv' - тренировочный и тестовый наборы данных.
'ML_1.ipynb' - блокнот с обучением моделей на 'train.csv'.
'test.ipynb' - блокнот для рассчета метрик по 'test.csv'.
'api_test.py' - файл теста API.

Признаки модели:
- Gender: ['Male', 'Female']
- Age: number
- Previously_Insured: [True, False]
- Vehicle_Age: ['< 1 Year', '1-2 Year', '> 2 Years']
- Vehicle_Damage: [True, False]

Нормализация данных по типу StandardScaler.

Технология в основе модели: Decision Tree.

API: http://77.222.54.240:8501/

## Авторы
- Горбунов Роман, R4160 (romangorbunov91)
- Иваненко Станислава, R4160 (smthCreate)
- Волынец Глеб, R4160 (glebvol12)
- Давыдов Игорь, R4197 (TriglCr)

## Reference
Описание признаков исходного датасета:
https://www.kaggle.com/datasets/annantkumarsingh/health-insurance-cross-sell-prediction-data/discussion/516324

KAGGLE Playground Series - Season 4, Episode 7: Binary Classification of Insurance Cross Selling
https://www.kaggle.com/competitions/playground-series-s4e7/overview

[AutoML Grand Prix] 1st Place Solution
https://www.kaggle.com/code/rohanrao/automl-grand-prix-1st-place-solution/notebook