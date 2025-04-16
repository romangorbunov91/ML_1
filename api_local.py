from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Загрузка модели из файла pickle
with open('d:\git\ITMO\ML\ML_1\clf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Счетчик запросов
request_count = 0
# Модель для валидации входных данных
class PredictionInput(BaseModel):
    Is_Male: float
    Age: float
    Previously_Insured: float
    Vehicle_Age: float
    Vehicle_Damage: float


@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    # Создание DataFrame из данных
    new_data = pd.DataFrame({
        'Is_Male': [input_data.Is_Male],
        'Age': [input_data.Age],
        'Previously_Insured': [input_data.Previously_Insured],
        'Vehicle_Age': [input_data.Vehicle_Age],
        'Vehicle_Damage': [input_data.Vehicle_Damage],

    })

    # Предсказание
    predictions = model.predict(new_data)

    # Преобразование результата в человеко-читаемый формат
    result = "Response negative" if predictions[0] == 1 else "Response positive"

    return {"prediction": result}

from typing import List

@app.post("/predict_model_batch")
def predict_model_batch(input_data_list: List[PredictionInput]):
    global request_count
    request_count += len(input_data_list)

    # Создание DataFrame из данных
    new_data = pd.DataFrame([item.model_dump() for item in input_data_list])

    # Предсказание
    predictions = model.predict(new_data.values)

    # Преобразование результата в человеко-читаемый формат
    results = ["Response negative" if pred == 1 else "Response positive" for pred in predictions]

    return {"predictions": results}



if __name__ == '__main__':

    uvicorn.run(app, host="127.0.0.1", port=5000)
#