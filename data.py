import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def data_prep(df):
    df['moving_avg'] = df.groupby('city')['temperature'].transform(lambda x: \
                                        x.rolling(window=30, min_periods=1).mean())
    seasonal_stats = df.groupby(['city', 'season'])['temperature']. \
                                                 agg(['mean', 'std']).reset_index()
    df = df.merge(seasonal_stats, on=['city', 'season'], suffixes=('', '_seasonal'))
    df['is_anomaly'] = (df['temperature'] < \
    (df['mean'] - 2 * df['std'])) | (df['temperature'] > (df['mean'] + 2 * df['std']))

    return df


def trend_calc(city_df, city):
    city_df = city_df[city_df['is_anomaly'] == False]
    city_df['timestamp'] = pd.to_datetime(city_df['timestamp'])

    # Подготовка данных
    X = city_df['timestamp'].astype('int64').values.reshape(-1, 1)
    y = city_df['temperature'].values

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X)

    # Оценка направления тренда
    trend = 'положительный' if model.coef_[0] > 0 else 'отрицательный'

    return (pred, f'Согласно представленным историческим данным наблюдений за температурой имеется {trend} тренд')


def city_stats(df, city_name):
    anomalies = df[df['is_anomaly'] == True]
    seasonal_profile = df.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()
    min_temp = df['temperature'].min()
    avg_temp = df['temperature'].mean()
    med_temp = df['temperature'].median()
    max_temp = df['temperature'].max()

    return {
        'city': city_name,
        'min_temperature': min_temp,
        'average_temperature': avg_temp,
        'median_temperature': avg_temp,
        'max_temperature': max_temp,
        'seasonal_profile': seasonal_profile,
        'anomalies': anomalies
    }


def compare_temperature(date, actual_temp, city, df):
    date = pd.to_datetime(date)

    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Фильтрация данных для конкретного города и указанной даты и без аномалий
    city_data = df[(df['city'] == city) &
                   (df['timestamp'].dt.month == date.month) &
                   (df['timestamp'].dt.day == date.day) &
                   (df['is_anomaly'] == False)]

    if city_data.empty:
        return "Нет данных для указанного города и даты."

    # Возьмём медианное значение исторической температуры на эту дату, если она не аномальная
    mean_temp = city_data['temperature'].median()

    # mean_temp = city_data['mean'].iloc[0] # менее точный вариант с полученным ранее сезонным средним
    std_temp = city_data['std'].iloc[0]

    # Вычисление отклонения фактической температуры от средней
    deviation = actual_temp - mean_temp

    # Определение, насколько фактическая температура отличается от исторической
    z_score = deviation / std_temp if std_temp != 0 else float('inf')

    # Вывод на основе Z-оценки
    if abs(z_score) < 1:
        assessment = "Температура близка к историческому среднему, не является аномалией."
    elif abs(z_score) < 2:
        assessment = '''Температура находится в пределах одного стандартного
    отклонения от среднего, что является нормальным размахом для таких данных.'''
    else:
        assessment = '''Температура считается аномальной относительно исторических
         данных — слишком холодной или слишком теплой.'''

    result = (
        f"Результат сравнения текущей температуры с историческими данными для города {city}\n"
        f"Фактическая температура: {actual_temp}°C\n"
        f"Средняя историческая температура для {city} на эту дату: {round(mean_temp, 2)}°C\n"
        f"Стандартное отклонение: {round(std_temp, 2)}°C\n"
        f"Отклонение фактической температуры: {round(deviation, 2)}°C\n"
        f"Z-оценка: {z_score:.2f}\n"
        f"Вывод: {assessment}\n"
        f"{'*' * 80}\n"
    )

    return result


def hist_plot(df, stats, city, pred):
    
    # Создаем график температуры
    fig = px.line(df, x='timestamp', y='temperature', labels={'temperature': 'Температура (°C)', 'timestamp': 'Год'},
         title=f'График температуры с аномальными значениями и линией тренда для города {city}')
    
    # Добавляем аномалии как отдельные точки
    fig.add_scatter(x=stats['anomalies']['timestamp'], y=stats['anomalies']['temperature'], mode='markers',
         name='Аномалии', marker=dict(color='red'))

    # Добавляем линию тренда
    fig.add_scatter(x=df['timestamp'], y=pred, mode='lines', 
                    name='Линия тренда', line=dict(color='black', width=2))

    fig.update_xaxes(
        tickformat="%Y",
        dtick="M12"
    )

    fig.update_layout(
        xaxis_title='Год',
        yaxis_title='Температура (°C)',
        legend=dict(x=0, y=1)
    )

    return fig