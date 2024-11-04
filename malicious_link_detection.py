# Импорт необходимых библиотек
import logging
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======================================
# Шаг 1: Сбор данных
# ======================================

def load_data(file_path):
    """
    Загрузка данных из CSV файла.

    Args:
        file_path (str): Путь к файлу CSV.

    Returns:
        pd.DataFrame: Загруженные данные.
    """
    try:
        data = pd.read_csv(file_path)
        required_columns = ['url', 'label']
        if not all(column in data.columns for column in required_columns):
            raise ValueError(f"Данные должны содержать столбцы: {required_columns}")
        logging.info("Данные успешно загружены.")
        return data
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        raise

# ======================================
# Шаг 2: Обработка данных
# ======================================

def extract_features(df):
    """
    Извлечение признаков из URL.

    Args:
        df (pd.DataFrame): Датафрейм с колонкой 'url'.

    Returns:
        pd.DataFrame: Датафрейм с новыми признаками.
    """
    df = df.copy()
    df['url_length'] = df['url'].apply(len)
    df['num_digits'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))
    df['num_special_chars'] = df['url'].apply(lambda x: len(re.findall(r'[\W]', x)))
    df['num_subdirs'] = df['url'].apply(lambda x: x.count('/'))
    df['num_params'] = df['url'].apply(lambda x: x.count('?') + x.count('&'))
    df['has_https'] = df['url'].apply(lambda x: int(x.startswith('https')))
    df['has_ip'] = df['url'].apply(lambda x: int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', x))))
    df['shortening_service'] = df['url'].apply(lambda x: int(bool(re.search(r'bit\.ly|t\.co|goo\.gl', x))))
    # Дополнительные признаки
    df['num_www'] = df['url'].apply(lambda x: x.count('www'))
    df['num_at'] = df['url'].apply(lambda x: x.count('@'))
    df['num_hyphen'] = df['url'].apply(lambda x: x.count('-'))
    return df

def preprocess_data(data):
    """
    Предобработка данных: извлечение признаков, обработка текстовых данных и разделение на X и y.

    Args:
        data (pd.DataFrame): Исходные данные.

    Returns:
        np.array, np.array: Массивы признаков и меток.
    """
    data = extract_features(data)
    feature_columns = ['url_length', 'num_digits', 'num_special_chars', 'num_subdirs',
                       'num_params', 'has_https', 'has_ip', 'shortening_service',
                       'num_www', 'num_at', 'num_hyphen']

    # Обработка текстовых данных с помощью TF-IDF
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(data['url'])

    # Нормализация числовых признаков
    scaler = StandardScaler()
    X_numeric = data[feature_columns]
    X_numeric_scaled = scaler.fit_transform(X_numeric)

    # Объединение всех признаков
    X = np.hstack((X_numeric_scaled, tfidf_matrix.toarray()))
    y = data['label'].values

    logging.info("Предобработка данных завершена.")
    return X, y, scaler, vectorizer, feature_columns

# ======================================
# Шаг 3: Создание и обучение моделей
# ======================================

def train_random_forest(X_train, y_train):
    """
    Обучение модели случайного леса с гиперпараметрической оптимизацией.

    Args:
        X_train (np.array): Признаки для обучения.
        y_train (np.array): Метки для обучения.

    Returns:
        RandomForestClassifier: Обученная модель.
    """
    oversampler = RandomOverSampler(random_state=42)

    # Параметры для поиска по сетке
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 10, 20],
        'rf__class_weight': [None, 'balanced']
    }

    pipeline = ImbPipeline([
        ('oversampler', oversampler),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logging.info(f"Лучшие параметры случайного леса: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_neural_network(X_train, y_train, X_val, y_val, input_dim):
    """
    Обучение нейронной сети.

    Args:
        X_train (np.array): Признаки для обучения.
        y_train (np.array): Метки для обучения.
        X_val (np.array): Признаки для валидации.
        y_val (np.array): Метки для валидации.
        input_dim (int): Размерность входных данных.

    Returns:
        keras.Model: Обученная модель.
    """
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    nn_model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = nn_model.fit(X_train, y_train, epochs=50, batch_size=64,
                           validation_data=(X_val, y_val),
                           class_weight=class_weight_dict,
                           callbacks=[early_stopping],
                           verbose=1)

    logging.info("Обучение нейронной сети завершено.")
    return nn_model, history

# ======================================
# Шаг 4: Оценка моделей
# ======================================

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Оценка модели и вывод метрик.

    Args:
        model: Обученная модель.
        X_test (np.array): Признаки для тестирования.
        y_test (np.array): Метки для тестирования.
        model_name (str): Название модели.
    """
    if isinstance(model, keras.Model):
        y_pred_proba = model.predict(X_test).flatten()
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_pred_proba >= 0.5).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f'{model_name} - Оценка модели:')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}\n')
    print('Отчет классификации:')
    print(classification_report(y_test, y_pred))

# ======================================
# Шаг 5: Реализация в реальном времени
# ======================================

def predict_url(model, url, scaler, vectorizer, feature_columns, threshold=0.5):
    """
    Предсказание для нового URL.

    Args:
        model: Обученная модель.
        url (str): URL для проверки.
        scaler: Объект StandardScaler для числовых признаков.
        vectorizer: Объект TfidfVectorizer для текстовых признаков.
        feature_columns (list): Список названий числовых признаков.
        threshold (float): Порог для классификации.

    Returns:
        str: Результат предсказания.
        float: Вероятность принадлежности к классу 1.
    """
    try:
        df = pd.DataFrame({'url': [url]})
        df = extract_features(df)
        X_numeric = df[feature_columns]
        X_numeric_scaled = scaler.transform(X_numeric)
        tfidf_features = vectorizer.transform(df['url']).toarray()
        X_features = np.hstack((X_numeric_scaled, tfidf_features))

        # Проверка наличия 'instagram.com' в URL
        if 'instagram.com' not in url:
            result = 'Предупреждение: обнаружена потенциальная угроза!'
            proba = 1.0  # Максимальная вероятность угрозы
            return result, proba

        # Если 'instagram.com' присутствует, используем модель для предсказания
        if isinstance(model, keras.Model):
            proba = model.predict(X_features)[0][0]
        else:
            proba = model.predict_proba(X_features)[0][1]

        if proba >= threshold:
            result = 'Предупреждение: обнаружена потенциальная угроза!'
        else:
            result = 'Ссылка безопасна.'
        return result, proba
    except Exception as e:
        logging.error(f"Ошибка при предсказании: {e}")
        return "Ошибка при обработке URL.", 0.0

# ======================================
# Шаг 6: Основной блок выполнения
# ======================================

if __name__ == "__main__":
    # Загрузка данных
    data = load_data('instagram_links.csv')

    # Предобработка данных
    X, y, scaler, vectorizer, feature_columns = preprocess_data(data)

    # Разделение данных на обучающую и тестовую выборки
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Разделение временной выборки на обучающую и валидационную
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp)

    # Обучение моделей
    rf_model = train_random_forest(X_train, y_train)
    nn_model, history = train_neural_network(X_train, y_train, X_val, y_val, input_dim=X_train.shape[1])

    # Оценка моделей
    evaluate_model(rf_model, X_test, y_test, model_name='Случайный лес')
    evaluate_model(nn_model, X_test, y_test, model_name='Нейронная сеть')

    # Проверка ссылок
    urls_to_check = [
        'https://instagram.com/username',
        'https://instagram.com/p/COdK1u1A5lS/',
        'http://fake-instagram.com/login',
        'http://instagram.com@phishing-site.com'
    ]

    print("\nПроверка ссылок:")
    for url in urls_to_check:
        result_rf, proba_rf = predict_url(rf_model, url, scaler, vectorizer, feature_columns)
        result_nn, proba_nn = predict_url(nn_model, url, scaler, vectorizer, feature_columns)
        print(f"URL: {url}")
        print(f"Случайный лес - Предсказание: {result_rf} (Вероятность угрозы: {proba_rf:.4f})")
        print(f"Нейронная сеть - Предсказание: {result_nn} (Вероятность угрозы: {proba_nn:.4f})\n")

    # Сохранение моделей и объектов предобработки
    joblib.dump(rf_model, 'rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    nn_model.save('nn_model.keras')

    logging.info("Модели и объекты предобработки сохранены.")
