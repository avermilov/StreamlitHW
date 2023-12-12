import pandas as pd
import catboost
import plotly.express as px
import streamlit as st
from PIL import Image

pd.options.plotting.backend = "plotly"
st.set_page_config(layout="wide")

PERCENTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
REMAP_DICT = {
    "GENDER": {"Мужчина": 1, "Женщина": 0},
    "IS_WORKING": {"Работает": 1, "Не работает": 0},
    "IS_PENSIONER": {"Пенсионер": 1, "Не пенсионер": 0},
    "HAS_FLAT": {"Имеет квартиру": 1, "Не имеет квартиру": 0},
    "CLOSED_CREDIT": {"Закрыл кредит": 1, "Не закрыл кредит": 0},
    "TARGET": {"Откликнулся": 1, "Не откликнулся": 0}
}


def get_corr_matrix(df):
    df_float = df.replace(REMAP_DICT).select_dtypes(include=["int", "float"])
    return df_float.corr()


MODEL = catboost.CatBoostClassifier()
MODEL.load_model("model_weights/model.weights")
SUCCESS_THRESHOLD = 0.5


def main():
    df = pd.read_csv("data/bank_dataset.csv").drop(columns=["ID_CLIENT", "ID_LOAN", "AGREEMENT_RK"])

    st.title("Отклик клиентов на маркетинговые предложения банка")
    st.subheader("Исследование данных и предсказание отклика")
    st.write("Данные - личные и экономические характеристики клиентов")
    st.image(Image.open("images/bank_welcome.jpg"))

    eda_tab, prediction_tab = st.tabs([":mag: Анализ данных", ":crystal_ball: Предсказание отклика"])

    with eda_tab:
        st.subheader("EDA: Анализируем данные")

        st.subheader("Статистические характеристики: числовые признаки")
        num_desc = df.describe(include=["int", "float"], percentiles=PERCENTILES)
        st.dataframe(num_desc, height=(num_desc.shape[0] + 1) * 35 + 3)

        # HISTOGRAMS
        st.subheader("Графики распределения: числовые признаки")
        float_feature = st.selectbox("Выберите вещественный признак:",
                                     options=list(df.select_dtypes(include=["int", "float"]).columns))
        st.plotly_chart(px.histogram(df, x=float_feature, color="TARGET",
                                     title=f"Распределение признака {float_feature}"
                                     ).update_xaxes(categoryorder="total descending"), use_container_width=True)
        st.markdown(
            """
            Выводы:
            - CHILD_TOTAL - Большинство не имеет детей или имеет одного ребенка.
            - DEPENDANTS - В основном у клиентов нет зависимых или же максимум есть один.
            - OWN_AUTO - Подавляющее большинство не имеют машины, а если имеют - то одну.
            - WORK_TIME - Большинство работает на текущем месте не более 8 лет.
            - PERSONAL_INCOME - Подавляющее большинство имеет зарплату от до 25 тысяч, но есть и очень редкие выбросы.
            - CREDIT - Медианная сумма кредита - около 12 тысяч, что совпадает с размером медианной зарплаты. Очень мало
              кто берет кредит больше, чем на 30 тысяч.
            - TERM - Подавляющее большинство берет кредит не более, чем на год.
            """)

        st.subheader("Статистические характеристики: категориальные признаки")
        str_desc = df.describe(include=["object"], percentiles=PERCENTILES)
        st.dataframe(str_desc, height=(str_desc.shape[0] + 1) * 35 + 3)

        st.subheader("Графики распределения: категориальные признаки")
        cat_feature = st.selectbox("Выберите категориальный признак:",
                                   options=list(df.select_dtypes(include="object").columns))
        st.plotly_chart(px.histogram(df, x=cat_feature, color="TARGET",
                                     title=f"Распределение признака {cat_feature}"
                                     ).update_xaxes(categoryorder="total descending"), use_container_width=True)
        st.markdown(
            """
            Выводы:
            - GENDER - В датасете больше мужчин, чем женщин (2 к 1).
            - EDUCATION - Подавляющее большинство клиентов имеет среднее или высшее образование. Очень мало клиентов
            без образования или с более высоким уровнем образования.
            - MARITAL_STATUS - Большинство состоит в браке.
            - IS_WORKING - Подавляющее большинство клиентов работает.
            - IS_PENSIONER - Подавляющее большинство клиентов не пенсионеры.
            - *_ADDRESS_PROVINCE - Более часто встречаются клиенты из Кемеровской области и Краснодарского края. 
            - HAS_FLAT - Большинство клиентов не имеет квартиры.
            - CLOSED_CREDIT - Закрывших и не закрывших кредит приблизительно поровну.
            - GEN_INDUSTRY - Более часто клиенты работают в сфере Торговли.
            - FAMILY_INCOME - Подавляющее большинство имеет семейны доход от 10 до 50 тысяч.
            """)

        # CORRELATION
        st.subheader("Корреляционный анализ")
        heatmap = get_corr_matrix(df)
        st.plotly_chart(px.imshow(heatmap))
        st.markdown(
            """
            Выводы:
            - Наибольшая положительная корреляция наблюдается между признаками AGE и IS_PENSIONER (в основном 
            пенсионерами становятся в пожилом возрасте), CHILD_TOTAL и DEPENDANTS (дети входят в число зависимых),
            CREDIT и FIRST_PAYMENT (чем больше кредит, тем больше обычно первый платеж), CREDIT и TERM (чем 
            больше кредит, тем длиннее срок погашения).
            - Наибольшая отрицательная корреляция наблюдается между IS_WORKING и IS_PENSIONER (обычно с возрастом
            количество работающих уменьшается), CLOSED_CREDIT и TERM (чем длиннее срок погашения, тем более вероятно,
            что кредит еще не закрыт), AGE и IS_WORKING (аналогично IS_WORKING и IS_PENSIONER).
            - Целевая переменная не коррелирует ни с каким одним признаком.
            """)

    with prediction_tab:
        st.subheader("Предсказываем отклик клиента")

        st.write("Личные данные клиента:")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Пол:", df.GENDER.unique())
            age = st.slider("Возраст:", min_value=1, max_value=100)
            postal_address_province = st.selectbox("Почтовый адрес области:", df.POSTAL_ADDRESS_PROVINCE.unique())

        with col2:
            education = st.selectbox("Образование:", df.EDUCATION.unique())
            marital_status = st.selectbox("Семейное положение:", df.MARITAL_STATUS.unique())
        with col3:
            child_total = st.slider("Количество детей:", min_value=0, max_value=10)
            dependants = st.slider("Количество иждивенцев:", min_value=0, max_value=10)
        st.divider()

        st.write("Экономические данные клиента:")
        col1, col2, col3 = st.columns(3)
        with col1:
            is_working = st.selectbox("Клиент работает?", df.IS_WORKING.unique())
            is_pensioner = st.selectbox("Клиент пенисонер?", df.IS_PENSIONER.unique())
            has_flat = st.selectbox("Клиент имеет квартиру?", df.HAS_FLAT.unique())

        with col2:
            own_auto = st.selectbox("Количество автомобилей в собственности:", df.OWN_AUTO.unique())
            family_income = st.selectbox("Суммарный доход на семью:", df.FAMILY_INCOME.unique())
            personal_income = st.slider("Личный доход:", min_value=1, max_value=250_000)

        with col3:
            gen_industry = st.selectbox("Сфера работы:", df.GEN_INDUSTRY.unique())

            gen_title = st.selectbox("Должность на работе:", df.GEN_TITLE.unique())
            job_dir = st.selectbox("Направление деятельности на работе:", df.JOB_DIR.unique())
            work_time = st.slider("Стаж работы на текущем месте (в месяцах):", min_value=1, max_value=720)
        st.divider()

        st.write("Информация о кредите клиента:")
        col1, col2 = st.columns(2)
        with col1:
            credit = st.slider("Сумма кредита:", min_value=1, max_value=300_000)
            term = st.slider("Срок кредита:", min_value=3, max_value=36)
        with col2:
            first_payment = st.slider("Первоначальный взнос по кредиту:", min_value=1, max_value=200_000)
            closed_credit = st.selectbox("Статус кредита:", df.CLOSED_CREDIT.unique())

        if col1.button("Предсказать отклик"):
            with st.spinner("Прогнозируем"):
                full_client_info = [{
                    "AGE": age, "GENDER": gender, "EDUCATION": education, "MARITAL_STATUS": marital_status,
                    "CHILD_TOTAL": child_total, "DEPENDANTS": dependants, "IS_WORKING": is_working,
                    "IS_PENSIONER": is_pensioner, "POSTAL_ADDRESS_PROVINCE": postal_address_province,
                    "HAS_FLAT": has_flat, "OWN_AUTO": own_auto, "CLOSED_CREDIT": closed_credit,
                    "GEN_INDUSTRY": gen_industry, "GEN_TITLE": gen_title,
                    "JOB_DIR": job_dir, "WORK_TIME": work_time, "FAMILY_INCOME": family_income,
                    "PERSONAL_INCOME": personal_income,
                    "CREDIT": credit, "TERM": term, "FIRST_PAYMENT": first_payment,
                }]
                client_df = pd.DataFrame(full_client_info).replace(REMAP_DICT)

                proba = MODEL.predict_proba(client_df)[0, 1]
                print(proba)
                if proba >= SUCCESS_THRESHOLD:
                    st.success("Клиент откликнется на маркетинговое предложение.")
                else:
                    st.error("Клиент не откликнется на маркетинговое предложение.")


if __name__ == "__main__":
    main()
