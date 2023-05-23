import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
import pickle
import numpy as np
import os
import joblib
import shap
import streamlit_shap
from streamlit_shap import st_shap
import matplotlib as plt
import xgboost
from xgboost import XGBClassifier

st.title('**Desenvolvimento e validação de um modelo de predição de mortalidade interpretável em pacientes gestantes, não-gestantes e puérperas hospitalizadas com SRAG, incluindo COVID19**')
st.subheader('_Interpretação do modelo utilizando valores Shapley (SHAP values)_')
st.markdown('Os uso de valores Shapley permite a interpretação local e global do modelo, ou seja, a identificação dos motivos que levaram o modelo a inferir cura ou óbito. Podemos avaliar o impacto de cada variável de forma individualizada para cada paciente bem como as interações entre as variáveis.')
st.markdown('**Por favor, responda as perguntas abaixo:**')


# widgets de entrada
Terceiro_Trimestre = st.selectbox('Está no terceiro trimestre gestacional?', options=['Sim', 'Não'])
Third_Trimester_pregnancy = 0
if Terceiro_Trimestre == 'Sim':
    Third_Trimester_pregnancy = 1
elif Terceiro_Trimestre == 'Não':
    Third_Trimester_pregnancy = 0

UTI = st.selectbox('Precisou de UTI?', options=['Sim', 'Não'])
UTI_Yes, UTI_No = 0, 0
if UTI == 'Sim':
    UTI_Yes = 1
else:
    UTI_No = 1

DESC_RESP = st.selectbox('Apresentou desconforto respiratório?', options=['Sim', 'Não'])
No_Distress_Respiratory = 0
if DESC_RESP == 'Não':
    No_Distress_Respiratory = 1
else:
    No_Distress_Respiratory = 0

DISPNEIA = st.selectbox('Apresentou dispneia?', options=['Sim', 'Não'])
Dispnea_No = 0
if DISPNEIA == 'Não':
    Dispnea_No = 1
else:
    Dispnea_No = 0

SATURACAO = st.selectbox('Apresentou saturação de oxigênio menor do que 95%?', options=['Sim', 'Não'])
No_O2_Sat_low_95 = 0
if SATURACAO == 'Não':
    No_O2_Sat_low_95 = 1
else:
    No_O2_Sat_low_95 = 0

etnia = st.selectbox('Tem etnia parda?', options=['Sim', 'Não'])
Brown_ethnicity = 0
if etnia == 'Sim':
    Brown_ethnicity = 1
elif etnia == 'Não':
    Brown_ethnicity = 0


SUPORT_VEN = st.selectbox('Utilizou suporte ventilatório?', options=['Sim, invasivo', 'Não'])
IOT_Yes, IOT_No = 0, 0
if SUPORT_VEN == 'Sim, invasivo':
    IOT_Yes = 1
else:
    IOT_No = 1

Idade = st.selectbox('Tem entre 21 e 30 anos de idade?', options=['Sim', 'Não'])
Age_21_30_Years_old = 0
if Idade == 'Sim':
    Age_21_30_Years_old = 1
else:
    Age_21_30_Years_old = 0

CS_ESCOL_N = st.selectbox('Possui ensino superior?', options=['Sim', 'Não'])
Superior_Scolarity = 0
if CS_ESCOL_N == 'Sim':
    Superior_Scolarity = 1
else:
    Superior_Scolarity = 0

PERD_OLFT = st.selectbox('Paciente apresentou perda do olfato?', options=['Sim', 'Não'])
Olfaction_Loss_Yes = 0
if PERD_OLFT == 'Sim':
    Olfaction_Loss_Yes = 1
else:
    Olfaction_Loss_Yes = 0

DIARREIA = st.selectbox('Paciente apresentou diarreia?', options=['Sim', 'Não'])
Diarrhea_Yes = 0
if DIARREIA == 'Sim':
    Diarrhea_Yes = 1
else:
    Diarrhea_Yes = 0

TOSSE = st.selectbox('Paciente apresentou tosse?', options=['Sim', 'Não'])
Cough_No = 0
if TOSSE == 'Não':
    Cough_No = 1
else:
    Cough_No = 0

CLASSI_FIN = st.selectbox('Sabe qual foi o agente causador?', options=['Sim', 'Não'])
Not_Specificate_Cause = 0
if CLASSI_FIN == 'Não':
    Not_Specificate_Cause = 1
else:
    Not_Specificate_Cause = 0



st.subheader('**Agora que temos as variáveis, faremos a classificação do paciente nas classes: cura ou óbito**')

data = pd.DataFrame([{'Brown_ethnicity': Brown_ethnicity, 'Superior_Scolarity': Superior_Scolarity, 'Cough_No': Cough_No, 'Dispnea_No': Dispnea_No, 'No_Distress_Respiratory': No_Distress_Respiratory, 'No_O2_Sat_low_95': No_O2_Sat_low_95, 'Diarrhea_Yes': Diarrhea_Yes, 'Olfaction_Loss_Yes': Olfaction_Loss_Yes, 'IOT_Yes': IOT_Yes, 'IOT_No': IOT_No, 'UTI_Yes': UTI_Yes, 'UTI_No': UTI_No, 'Not_Specificate_Cause': Not_Specificate_Cause, 'Age_21_30_Years_old': Age_21_30_Years_old, 'Third_Trimester_pregnancy': Third_Trimester_pregnancy}])

# carregando o modelo treinado
model = joblib.load('C:/Users/danip/venv/app_karla/RandomForest.pkl')
# realizando as predições
result = model.predict(data)
proba = model.predict_proba(data)

if st.button('Qual é o provável resultado?'):
    if result == 0:
        st.success('Cura')
        st.write('Com probabilidade de: {}%'.format(proba[:,0]*100))
    else:
        st.success('Óbito')
        st.write('Com probabilidade de: {}%'.format(proba[:,1]*100))
    
    

# utilizando shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data)
shap.initjs()

st.subheader('Interpretando o modelo')
st.markdown('***Classe 0: CURA e Classe 1: ÓBITO***')
if st.button('Obtendo o impacto das variáveis nas possíveis saídas do modelo'):
    if result == 0:
        st.markdown('****Impacto médio na saída do modelo****')
        st_shap(shap.summary_plot(shap_values[0], data, plot_type='bar', plot_size=0.2))
        st.markdown('****Impacto das categorias específicas das variáveis na saída do modelo****')
        st_shap(shap.summary_plot(shap_values[0], data))
    else:
        st.markdown('****Impacto médio na saída do modelo****')
        st_shap(shap.summary_plot(shap_values[1], data, plot_type='bar', plot_size=0.2))
        st.markdown('****Impacto das categorias específicas das variáveis na saída do modelo****')
        st_shap(shap.summary_plot(shap_values[1], data))
if st.button('Quais variáveis mais impactaram na decisão do modelo?'):
    if result == 0:
        st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], data.iloc[0, :]))
    else:
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], data.iloc[0, :]))
if st.button('Por que o modelo tomou determinada decisão?'):
    st.markdown('**Gráfico de decisão**')
    if result == 0:
        st_shap(shap.decision_plot(explainer.expected_value[0], shap_values[0], feature_order="importance", feature_names=list(data.columns),
                   link='identity', highlight=0))
    else:
        st_shap(shap.decision_plot(explainer.expected_value[1], shap_values[1], feature_order="importance", feature_names=list(data.columns),
                   link='logit', highlight=0))
