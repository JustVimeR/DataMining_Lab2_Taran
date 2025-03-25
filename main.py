import pandas as pd
import numpy as np
import os

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px

# ЗАВАНТАЖЕННЯ ТА ОБРОБКА ДАНИХ

DATA_PATH = "./ratings_Electronics (1).csv"
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Файл {DATA_PATH} не знайдено.")

# Читаємо CSV БЕЗ заголовка та присвоюємо імена стовпчикам
df_full = pd.read_csv(
    DATA_PATH,
    header=None,
    names=['userId', 'productId', 'rating', 'timestamp']
)

# Вибірка до 1,000,000 рядків
if len(df_full) > 1000000:
    df_sample = df_full.sample(n=1000000, random_state=42)
else:
    df_sample = df_full.copy()

# Перетворимо "timestamp" на дату (Unix time → datetime)
df_sample['date'] = pd.to_datetime(df_sample['timestamp'], unit='s')
df_sample['year_month'] = df_sample['date'].dt.to_period('M').astype(str)  # рік-місяць у рядковому форматі
df_sample.drop(columns='timestamp', inplace=True)


# EDA: ПЕРШИЙ ПОГЛЯД

unique_users = df_sample['userId'].nunique()
unique_items = df_sample['productId'].nunique()
print(f"Вибірка: {len(df_sample)} рядків")
print(f"Унікальних користувачів: {unique_users}")
print(f"Унікальних товарів: {unique_items}")

mean_rating = df_sample['rating'].mean()
print(f"Середній рейтинг: {mean_rating:.2f}")

# Додатковий EDA: Розподіл рейтингів
rating_distribution = df_sample['rating'].value_counts().sort_index()
rating_distribution_fig = px.bar(
    rating_distribution,
    x=rating_distribution.index,
    y=rating_distribution.values,
    labels={'x': 'Rating', 'y': 'Count'},
    title='Розподіл Рейтингів',
    template='plotly_white'
)

# Топ-10 користувачів за кількістю оцінок
top_users_by_ratings = df_sample['userId'].value_counts().nlargest(10)
top_users_fig = px.bar(
    top_users_by_ratings,
    x=top_users_by_ratings.values,
    y=top_users_by_ratings.index,
    orientation='h',
    labels={'x': 'Number of Ratings', 'y': 'UserId'},
    title='Топ-10 Користувачів за Кількістю Оцінок',
    template='plotly_white'
)

# Додатковий EDA: Розподіл кількості оцінок на товар
top_products_by_ratings = df_sample['productId'].value_counts().nlargest(10)
top_products_fig = px.bar(
    top_products_by_ratings,
    x=top_products_by_ratings.values,
    y=top_products_by_ratings.index,
    orientation='h',
    labels={'x': 'Number of Ratings', 'y': 'ProductId'},
    title='Топ-10 Товарів за Кількістю Оцінок',
    template='plotly_white'
)

# Додатковий EDA: Тренд оцінок по місяцях
ratings_by_month = df_sample.groupby('year_month').agg(
    total_ratings=('rating', 'count'),
    average_rating=('rating', 'mean')
).reset_index()

ratings_over_time_fig = px.line(
    ratings_by_month,
    x='year_month',
    y='total_ratings',
    title='Загальна Кількість Оцінок по Місяцях',
    labels={'year_month': 'Year-Month', 'total_ratings': 'Total Ratings'},
    template='plotly_white'
)

average_rating_over_time_fig = px.line(
    ratings_by_month,
    x='year_month',
    y='average_rating',
    title='Середній Рейтинг по Місяцях',
    labels={'year_month': 'Year-Month', 'average_rating': 'Average Rating'},
    template='plotly_white'
)


# СТВОРЕННЯ РЕКОМЕНДАЦІЙНОЇ СИСТЕМИ

surprise_reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_sample[['userId', 'productId', 'rating']], surprise_reader)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

svd_model = SVD(n_factors=50, random_state=42)
svd_model.fit(train_data)

predictions = svd_model.test(test_data)
rmse = accuracy.rmse(predictions, verbose=True)
mae = accuracy.mae(predictions, verbose=True)


# ГЕНЕРАЦІЯ РЕКОМЕНДАЦІЙ
def get_top_n_recommendations(algo, user_id, df_full, n=5):

    # Знайдемо всі товари, які користувач НЕ оцінив
    user_data = df_full[df_full['userId'] == user_id]
    rated_items = set(user_data['productId'].unique())
    all_items = set(df_full['productId'].unique())
    unrated_items = list(all_items - rated_items)

    # Робимо передбачення рейтингу
    predictions = []
    for item_id in unrated_items:
        pred = algo.predict(str(user_id), str(item_id))
        predictions.append((item_id, pred.est))

    # Сортуємо за очікуваним рейтингом (спадно)
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:n]


# Приклад: Рекомендації для випадкового користувача
sample_user_id = df_sample['userId'].sample(1, random_state=42).iloc[0]
sample_user_recommendations = get_top_n_recommendations(svd_model, sample_user_id, df_sample, n=5)
print(f"Рекомендації для користувача {sample_user_id}: {sample_user_recommendations}")


# Time Series Line Chart
def generate_time_series_fig(df):

    ratings_over_time = df.groupby('year_month').agg(
        total_ratings=('rating', 'count'),
        average_rating=('rating', 'mean')
    ).reset_index()

    fig = px.line(
        ratings_over_time,
        x='year_month',
        y=['total_ratings', 'average_rating'],
        title='Тренд Оцінок по Місяцях',
        labels={'year_month': 'Year-Month'},
        template='plotly_white',
        markers=True
    )

    fig.update_layout(
        xaxis_title="Year-Month",
        yaxis_title="Metrics",
        legend_title="Metrics",
        hovermode="x unified"
    )

    return fig


ratings_trend_fig = generate_time_series_fig(df_sample)

external_stylesheets = ["https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

unique_user_ids = df_sample['userId'].unique()
unique_user_ids = unique_user_ids.tolist()

app.layout = html.Div([
    html.Div([
        html.H1("📊 Recommender System Dashboard", style={
            'textAlign': 'center',
            'fontSize': '36px',
            'color': '#333',
            'marginBottom': '30px'
        }),

        html.Div([
            html.H2("🔍 Основні Метрики Моделі"),
            html.Div([
                html.P(f"✅ RMSE на тесті: {rmse:.4f}"),
                html.P(f"✅ MAE на тесті: {mae:.4f}"),
                html.P(f"👥 Кількість користувачів: {unique_users}"),
                html.P(f"📦 Кількість товарів: {unique_items}"),
                html.P(f"⭐ Середній рейтинг: {mean_rating:.2f}")
            ], className='card')
        ], className='section'),

        html.Div([
            html.H2("🎯 Отримати Рекомендації"),
            html.Div([
                html.P("Введіть або виберіть UserId:"),
                dcc.Input(id='input-user', type='text', placeholder='UserId', value=str(sample_user_id), className='input'),
                dcc.Dropdown(id='user-suggestions', options=[], placeholder='Виберіть UserId', style={'display': 'none'}),
                html.Button("🔎 Отримати Рекомендації", id='recommend-button', n_clicks=0, className='button'),
                html.Div(id='recommendation-result', className='card', style={'marginTop': '15px'})
            ])
        ], className='section'),

        html.Div([
            html.H2("📈 Візуалізації"),
            dcc.Graph(id='rating-distribution', figure=rating_distribution_fig),
            dcc.Graph(id='top-users', figure=top_users_fig),
            dcc.Graph(id='top-products', figure=top_products_fig)
        ], className='section'),

        html.Div([
            html.H2("🕒 Тренд Оцінок по Місяцях"),
            dcc.Graph(id='time-series-chart', figure=ratings_trend_fig)
        ], className='section'),
    ], className='container')
], style={'fontFamily': "'Inter', sans-serif", 'backgroundColor': '#f4f7fa', 'padding': '20px'})

@app.callback(
    [Output('user-suggestions', 'options'),
     Output('user-suggestions', 'style')],
    Input('input-user', 'value')
)
def update_user_suggestions(input_value):
    if len(input_value) >= 3:
        # Фільтруємо userIds, що містять введений текст (case-insensitive)
        matching_user_ids = [uid for uid in unique_user_ids if input_value.lower() in uid.lower()]
        # Обмежимо кількість пропозицій, наприклад, до 10
        matching_user_ids = matching_user_ids[:10]
        options = [{'label': uid, 'value': uid} for uid in matching_user_ids]
        if options:
            return options, {'width': '300px', 'marginTop': '5px', 'display': 'block'}
    # Якщо менше 3 символів або нема пропозицій
    return [], {'width': '300px', 'marginTop': '5px', 'display': 'none'}

@app.callback(
    Output('input-user', 'value'),
    Input('user-suggestions', 'value'),
    State('input-user', 'value')
)
def set_input_value(selected_user, current_input):
    if selected_user:
        return selected_user
    return current_input

@app.callback(
    Output('recommendation-result', 'children'),
    Input('recommend-button', 'n_clicks'),
    State('input-user', 'value')
)
def update_recommendations(n_clicks, user_id_value):
    if n_clicks > 0:
        try:
            user_id_value = str(user_id_value)
            if user_id_value not in df_sample['userId'].unique():
                return html.P(f"Користувач {user_id_value} не знайдений.", style={'color': 'red'})
            top_n_recs = get_top_n_recommendations(svd_model, user_id_value, df_sample, n=5)
            if len(top_n_recs) == 0:
                return f"Користувач {user_id_value} уже оцінив усі товари або не знайдено даних."
            else:
                return [
                    html.P(f"Рекомендації для користувача {user_id_value}:"),
                    html.Ul([html.Li(f"Товар: {prod}, очікуваний рейтинг: {est:.2f}") for prod, est in top_n_recs])
                ]
        except Exception as e:
            return f"Помилка при обчисленні рекомендацій: {e}"
    return ""


if __name__ == '__main__':
    app.run(debug=True)
