import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


st.set_page_config(page_title='Подборка вин', layout='wide')
st.title('Подборка хороших вин')
st.write('Хорошим считается вино с экспертной оценкой от 7')

model = load_model()
test = load_test()
X_test = test.drop(columns='target')
y_test = test['target']
p_test = test_proba(model, X_test)
config = json.loads((APP_DIR / 'config.json').read_text())

threshold = st.sidebar.slider('Порог отбора', 0.0, 1.0, float(config['threshold']), step=0.001, format='%.3f')
st.sidebar.write('Чем выше порог - тем строже отбор')
one_tab, metrics_tab, batch_tab, coef_tab = st.tabs(
    ['Одно вино', 'Модельные метрики', 'CSV с винами', 'Что выучила модель'])

labels = {'fixed_acidity': 'Нелетучие кислоты',
          'volatile_acidity': 'Летучие кислоты',
          'citric_acid': 'Лимонная кислота',
          'residual_sugar': 'Остаточный сахар',
          'chlorides': 'Хлориды',
          'free_sulfur_dioxide': 'Свободный диоксид серы',
          'total_sulfur_dioxide': 'Общий диоксид серы',
          'density': 'Плотность',
          'pH': 'pH',
          'sulphates': 'Сульфаты',
          'alcohol': 'Алкоголь',
          'is_red': 'Красное вино'}

with one_tab:
    st.write('Введите параметры:')
    values = {}
    with st.form('one_wine'):
        columns = st.columns(3)
        for i, feat in enumerate(X_test.columns):
            with columns[i % 3]:
                if feat == 'is_red':
                    values[feat] = int(st.toggle('Красное вино', value=bool(X_test[feat].median())))
                else:
                    values[feat] = st.number_input(
                        labels[feat], min_value=float(X_test[feat].min()),
                        max_value=float(X_test[feat].max()), value=float(X_test[feat].median()),
                        step=0.00001 if feat == 'density' else 0.001, format='%.5f', help=feat)
        submitted = st.form_submit_button('Оценить вино')
    if submitted:
        row = pd.DataFrame([values], columns=X_test.columns)
        st.session_state['wine_probability'] = float(model.predict_proba(row)[0, 1])
    if 'wine_probability' in st.session_state:
        probability = st.session_state['wine_probability']
        st.metric('Вероятность хорошего вина', str(round(probability * 100, 2)) + '%')
        st.write('Решение: добавить в подборку' if probability >= threshold else 'Решение: пропустить')

with metrics_tab:
    pred = p_test >= threshold
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred)
    columns = st.columns(3)
    for col, label, value in zip(columns, ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        col.metric(label, round(value, 4))
    st.write('При повышении порога подборка станет меньше, то есть хорошие вина чаще будут пропускаться, '
             'но отбор станет точнее. А при понижении порога попадет больше вин, '
             'в том числе и посредственные')

    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Остальные', 'Хорошие']).plot(
        ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set(xlabel='Предсказание', ylabel='Истинный класс', title='Матрица ошибок')
    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].scatter(fp / (fp + tn), recall, color='red', zorder=3, label='Текущий порог')
    axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC')
    prec, rec, _ = metrics.precision_recall_curve(y_test, p_test)
    axes[2].plot(rec, prec)
    axes[2].axhline(y_test.mean(), ls='--', color='gray', label='Случайная модель')
    # При пустой подборке precision не определен; на конце PR принято ставить 1
    axes[2].scatter(recall, precision if pred.any() else 1.0,
                    color='red', zorder=3, label='Текущий порог')
    axes[2].set(xlabel='Recall', ylabel='Precision', title='PR')
    axes[1].legend()
    axes[2].legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if not pred.any():
        st.caption('Ни одно вино не выбрано: precision принят за 0')
    st.caption('Порог здесь меняет только решение, итоговый порог выбран по валидации')

with batch_tab:
    st.write('Загрузите CSV с теми же признаками без target, для отбора будет использоваться порог слева')
    st.caption('Столбцы: ' + ', '.join(X_test.columns))
    uploaded = st.file_uploader('CSV с признаками', type=['csv'])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = X_test.columns.difference(batch.columns)
            if len(missing):
                st.error('Не хватает столбцов: ' + ', '.join(missing))
            elif batch.empty:
                st.error('В файле нет строк с винами')
            else:
                features = batch[X_test.columns].apply(pd.to_numeric, errors='coerce')
                if not np.isfinite(features.to_numpy()).all():
                    st.error('В признаках есть пропуски или нечисловые значения')
                elif not features['is_red'].isin([0, 1]).all():
                    st.error('В столбце is_red должны быть только 0 и 1')
                else:
                    batch['probability'] = model.predict_proba(features)[:, 1]
                    batch['selected'] = (batch['probability'] >= threshold).astype(int)
                    st.dataframe(batch, hide_index=True)
                    st.download_button('Скачать результат',
                                       batch.to_csv(index=False).encode('utf-8'),
                                       file_name='wine_predictions.csv', mime='text/csv')
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            st.error('Не удалось прочитать CSV')

with coef_tab:
    coef = pd.Series(model.named_steps['logisticregression'].coef_[0],
                     index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    coef.plot.barh(ax=ax, color=np.where(coef > 0, '#2d8768', '#bd5366'))
    ax.axvline(0, color='black', lw=0.8)
    ax.set(xlabel='Коэффициент после StandardScaler', ylabel='Признак')
    st.pyplot(fig)
    plt.close(fig)
    st.write('Сильнее всего влияет в положительную сторону (за хорошее вино): ' + ', '.join(coef.nlargest(3).index))
    st.write('Сильнее всего против: ' + ', '.join(coef.nsmallest(3).index))