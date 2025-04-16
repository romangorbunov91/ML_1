import pickle
import numpy as np
import streamlit as st
import requests
from requests.exceptions import ConnectionError
from sklearn.preprocessing import StandardScaler

# Загрузка scaler для масштабирования
def load_scaler():
    try:
        with open('sc_model.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except Exception as e:
        st.error(f"Error loading the scaler: {e}")
        return None

ip_api = "ml-api"
port_api = "5000"

# Заголовок приложения
st.title("Insurance Selling Prediction")

# Ввод данных
st.write("Enter the client details:")

# Выпадающее меню для выбора класса билета
is_male = st.selectbox("Gender", ["Male","Female"])

# Текстовое поле для ввода возраста с проверкой на число
age = st.text_input("Age", value=10)
if not age.isdigit():
    st.error("Please enter a valid number for Age.")


previously_insured = st.selectbox("Previously Insured", [True,False])
vehicle_age = st.selectbox("Vehicle age", ["< 1 Year","1-2 Year","> 2 Years"])
vehicle_damage = st.selectbox("Vehicle Damage", [True,False])


# Кнопка для отправки запроса
if st.button("Predict"):
    # Проверка, что все поля заполнены
    if age.isdigit():
        # Подготовка данных для отправки
        data = {
            "Is_Male": int( 1 if is_male == "Male" else 0),  # Нормализованные данные
            "Age": int(age),
            "Previously_Insured": int( 1 if previously_insured else 0),
            "Vehicle_Age": int(0 if vehicle_age == "< 1 Year" else (1 if vehicle_age == "1-2 Year" else 2)),
            "Vehicle_Damage": int( 1 if vehicle_damage else 0),
        }

        # Преобразуем данные в формат numpy для применения скалера
        data_array = np.array([list(data.values())])

        # Загружаем scaler
        scaler = load_scaler()
        if scaler:
            # Масштабируем данные
            scaled_data = scaler.transform(data_array)

            # Преобразуем масштабированные данные обратно в список для отправки
            scaled_data_dict = {
                "Is_Male": scaled_data[0][0],
                "Age": scaled_data[0][1],
                "Previously_Insured": scaled_data[0][2],
                "Vehicle_Age": scaled_data[0][3],
                "Vehicle_Damage": scaled_data[0][4],
            }
            try:
                # Отправка запроса к Flask API
                response = requests.post(f"http://{ip_api}:{port_api}/predict_model", json=scaled_data_dict)

                # Проверка статуса ответа
                if response.status_code == 200:
                    prediction = response.json()["prediction"]
                    st.success(f"Prediction: {prediction}")
                else:
                    st.error(f"Request failed with status code {response.status_code}")
            except ConnectionError as e:
                st.error(f"Failed to connect to the server")
        else:
            st.error("Failed to load scaler.")

    else:
        st.error("Please fill in all fields with valid numbers.")