import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Вспомогательная функция для генерации случайных данных
def generate_electron_data(num_samples=100):
    np.random.seed(None)  # сброс текущего seed для уникальности данных
    energies = np.random.normal(loc=5, scale=2, size=num_samples)
    momenta = np.random.normal(loc=3, scale=1, size=num_samples)
    return pd.DataFrame({'energy': energies, 'momentum': momenta})

# Основной поток приложения
st.title('Интерактивный анализ кластеризации электронов')

# Кнопка для генерации новых данных
generate_new_data = st.button('Сгенерировать новые данные')

# Определяем источник данных
if generate_new_data or ('data' not in st.session_state):
    df = generate_electron_data()  # генерировать новые данные
    st.session_state.data = df  # сохраним в session state для повторного использования
else:
    df = st.session_state.data  # используем старые данные

# Настройки интерфейса
n_clusters = st.slider("Количество кластеров:", min_value=2, max_value=10, value=3)

# K-means кластеризация
model = KMeans(n_clusters=n_clusters)
labels = model.fit_predict(df.values)

# Отображение графика
fig, ax = plt.subplots()
scatter = ax.scatter(df['energy'], df['momentum'], c=labels, cmap='viridis', s=50)
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.set_xlabel('Энергия')
ax.set_ylabel('Импульс')
ax.add_artist(legend1)
st.pyplot(fig)