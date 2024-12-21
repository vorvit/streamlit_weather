import asyncio
from datetime import date
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from client import get_temperature_async
from data import data_prep, compare_temperature, hist_plot, city_stats, trend_calc

CITY_COORDINATES = {
    'New York': (40.7128, -74.0060),
    'London': (51.5074, -0.1278),
    'Paris': (48.8566, 2.3522),
    'Tokyo': (35.6895, 139.6917),
    'Moscow': (55.7558, 37.6173),
    'Sydney': (-33.8688, 151.2093),
    'Berlin': (52.5200, 13.4050),
    'Beijing': (39.9042, 116.4074),
    'Rio de Janeiro': (-22.9068, -43.1729),
    'Dubai': (25.276987, 55.296249),
    'Los Angeles': (34.0522, -118.2437),
    'Singapore': (1.3521, 103.8198),
    'Mumbai': (19.0760, 72.8777),
    'Cairo': (30.0444, 31.2357),
    'Mexico City': (19.4326, -99.1332)
}

PAGE_TITLE = "Weather"
PAGE_ICON_PATH = 'bounty.jpg'
ZOOM_LEVEL = 4.5
RADIUS = 30000

today = date.today()

def set_page_config():
    image = Image.open(PAGE_ICON_PATH)
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title=PAGE_TITLE,
        page_icon=image,
    )


def render_main_page():
    image = Image.open(PAGE_ICON_PATH)
    st.write('''# Анализ текущей температуры в выбранном городе''')
    st.image(image)


def process_file_upload():
    st.header("Загрузите данные наблюдений")
    uploaded_file = st.file_uploader("Загрузите Ваш датасет с температурой", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return data_prep(df)
    st.write("Пожалуйста, загрузите CSV-файл.")
    return None


def select_city_mode(cities):
    st.header("Выберите город, или список городов из датасета")
    return st.radio("Выберите один из вариантов:", ['Город', 'Список городов'])


def render_city_map(city, latitude, longitude):
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=[{'latitude': latitude, 'longitude': longitude}],
        get_position='[longitude, latitude]',
        get_color='[200, 30, 0, 160]',
        get_radius=RADIUS,
    )
    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=ZOOM_LEVEL,
        pitch=0,
    )
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/streets-v11',
        tooltip={"text": city}
    )
    st.pydeck_chart(r)


def handle_city_selection(df, city):
    latitude, longitude = CITY_COORDINATES[city]
    st.write(f'Координаты {city}: широта {latitude}, долгота {longitude}')
    render_city_map(city, latitude, longitude)
    city_df = df[df['city'] == city]
    st.header(f"Исторические данные температуры в {city}")
    st.write(city_df)
    stats = city_stats(city_df, city)

    st.write(f"Минимальная температура: {stats['min_temperature']:.2f}")
    st.write(f"Средняя температура: {stats['average_temperature']:.2f}")
    st.write(f"Медианная температура: {stats['median_temperature']:.2f}")
    st.write(f"Максимальная температура: {stats['max_temperature']:.2f}")

    pred, trend = trend_calc(city_df, city)
    st.write(trend)

    st.header("Профиль сезонности")
    st.dataframe(stats['seasonal_profile'])

    st.plotly_chart(hist_plot(city_df, stats, city, pred))


def fetch_and_display_temp(city, api_key, df):
    temp = asyncio.run(get_temperature_async(city, api_key))
    if isinstance(temp, dict) and temp.get("cod") != 200:
        error_message = temp.get("message", "Неизвестная ошибка")
        st.error(f"Ошибка получения температуры для {city}: {error_message}")
    else:
        actual_temp_output = f"## Текущая температура в {city}: {temp}°C"
        st.write(actual_temp_output)
        st.text(compare_temperature(today, temp, city, df))


def handle_city_temperature(cities, city_mode, api_key, df, city=None):
    if api_key:
        with st.form(key='weather'):
            if st.form_submit_button("Получить температуру на сегодня"):
                if city_mode == 'Город':
                    fetch_and_display_temp(city, api_key, df)
                else:
                    for city in cities:
                        fetch_and_display_temp(city, api_key, df)
    else:
        st.write("Пожалуйста, введите Ваш API-ключ")


def main():
    set_page_config()
    render_main_page()
    df = process_file_upload()
    if df is not None:
        cities = df['city'].unique()
        city_mode = select_city_mode(cities)
        city = None
        if city_mode == 'Город':
            city = st.selectbox("## Выберите город из списка:", cities)
            handle_city_selection(df, city)
        else:
            st.write("Вы выбрали Список городов")
        api_key = st.text_input("Введите Ваш API-ключ:", type="password")
        handle_city_temperature(cities, city_mode, api_key, df, city)


if __name__ == "__main__":
    main()