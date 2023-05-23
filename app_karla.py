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

# # widgets de entrada
# Primeiro_Trimestre = st.selectbox('Está no primeiro trimestre gestacional?', options=['Sim', 'Não'])
# Primeiro_Trimestre_1, Primeiro_Trimestre_2 = 0, 0
# if Primeiro_Trimestre == 'Sim':
#     Primeiro_Trimestre_1 = 1
# else:
#     Primeiro_Trimestre_2 = 1

# Segundo_Trimestre = st.selectbox('Está no segundo trimestre gestacional?', options=['Sim', 'Não'])
# Segundo_Trimestre_1, Segundo_Trimestre_2 = 0, 0
# if Segundo_Trimestre == 'Sim':
#     Segundo_Trimestre_1 = 1
# else:
#     Segundo_Trimestre_2 = 1

# Terceiro_Trimestre = st.selectbox('Está no terceiro trimestre gestacional?', options=['Sim', 'Não'])
# Terceiro_Trimestre_1, Terceiro_Trimestre_2 = 0, 0
# if Terceiro_Trimestre == 'Sim':
#     Terceiro_Trimestre_1 = 1
# else:
#     Terceiro_Trimestre_2 = 1

# UTI = st.selectbox('Precisou de UTI?', options=['Sim', 'Não'])
# UTI_1, UTI_2 = 0, 0
# if UTI == 'Sim':
#     UTI_1 = 1
# else:
#     UTI_2 = 1

# DESC_RESP = st.selectbox('Apresentou desconforto respiratório?', options=['Sim', 'Não'])
# DESC_RESP_1, DESC_RESP_2 = 0, 0
# if DESC_RESP_1 == 'Sim':
#     DESC_RESP_1 = 1
# else:
#     DESC_RESP_2 = 1

# DISPNEIA = st.selectbox('Apresentou dispneia?', options=['Sim', 'Não'])
# DISPNEIA_1, DISPNEIA_2 = 0, 0
# if DISPNEIA == 'Sim':
#     DISPNEIA_1 = 1
# else:
#     DISPNEIA_2 = 1

# SATURACAO = st.selectbox('Apresentou saturação de oxigênio menor do que 95%?', options=['Sim', 'Não'])
# SATURACAO_1, SATURACAO_2 = 0, 0
# if SATURACAO == 'Sim':
#     SATURACAO_1 = 1
# else:
#     SATURACAO_2 = 1

# HEMATOLOGI = st.selectbox('Paciente apresentava doença hematológica crônica?', options=['Sim', 'Não'])
# HEMATOLOGI_1, HEMATOLOGI_2 = 0, 0
# if HEMATOLOGI == 'Sim':
#     HEMATOLOGI_1 = 1
# else:
#     HEMATOLOGI_2 = 1

# CS_RACA = st.selectbox('Qual é a etnia do paciente?', options=['Branca', 'Preta', 'Amarela', 'Parda', 'Indígena'])
# CS_RACA_1, CS_RACA_2, CS_RACA_3, CS_RACA_4, CS_RACA_5 = 0, 0, 0, 0, 0
# if CS_RACA == 'Branca':
#     CS_RACA_1 = 1
# if CS_RACA == 'Preta':
#     CS_RACA_2 = 1
# if CS_RACA == 'Amarela':
#     CS_RACA_3 = 1
# if CS_RACA == 'Parda':
#     CS_RACA_4 = 1
# else:
#     CS_RACA_5 = 1

# PNEUMOPATI = st.selectbox('Paciente apresentava pneumopatia crônica?', options=['Sim', 'Não'])
# PNEUMOPATI_1, PNEUMOPATI_2 = 0, 0
# if PNEUMOPATI == 'Sim':
#     PNEUMOPATI_1 = 1
# else:
#     PNEUMOPATI_2 = 1

# ANTIVIRAL = st.selectbox('Paciente fez uso de algum antiviral?', options=['Sim', 'Não'])
# ANTIVIRAL_1, ANTIVIRAL_2 = 0, 0
# if ANTIVIRAL == 'Sim':
#     ANTIVIRAL_1 = 1
# else:
#     ANTIVIRAL_2 = 1

# TOMO_RES = st.selectbox('Qual foi o resultado da tomografia?', options=['Tipico COVID-19', 'Indeterminado COVID-19', 'Atípico COVID-19', 'Negativo para Pneumonia', 'Outro', 'Não realizado'])
# TOMO_RES_1, TOMO_RES_2, TOMO_RES_3, TOMO_RES_4, TOMO_RES_5, TOMO_RES_6 = 0, 0, 0, 0, 0, 0
# if TOMO_RES == 'Tipico COVID-19':
#     TOMO_RES_1 = 1
# if TOMO_RES == 'Indeterminado COVID-19':
#     TOMO_RES_2 = 1
# if TOMO_RES == 'Atípico COVID-19':
#     TOMO_RES_3 = 1
# if TOMO_RES == 'Negativo para Pneumonia':
#     TOMO_RES_4 = 1
# if TOMO_RES == 'Outro':
#     TOMO_RES_5 = 1
# else:
#     TOMO_RES_6 = 1

# RAIOX_RES = st.selectbox('Qual foi o resultado do raio-x?', options=['Normal', 'Infiltrado intersticial', 'Consolidação', 'Misto', 'Outro', 'Não realizado'])
# RAIOX_RES_1, RAIOX_RES_2, RAIOX_RES_3, RAIOX_RES_4, RAIOX_RES_5, RAIOX_RES_6 = 0, 0, 0, 0, 0, 0
# if RAIOX_RES == 'Normal':
#     RAIOX_RES_1 = 1
# if RAIOX_RES == 'Infiltrado intersticial':
#     RAIOX_RES_2 = 1
# if RAIOX_RES == 'Consolidação':
#     RAIOX_RES_3 = 1
# if RAIOX_RES == 'Misto':
#     RAIOX_RES_4 = 1
# if RAIOX_RES == 'Outro':
#     RAIOX_RES_5 = 1
# else:
#     RAIOX_RES_6 = 1

# SUPORT_VEN = st.selectbox('Utilizou suporte ventilatório?', options=['Sim, invasivo', 'Sim, não invasivo', 'Não'])
# SUPORT_VEN_1, SUPORT_VEN_2, SUPORT_VEN_3 = 0, 0, 0
# if SUPORT_VEN == 'Sim, invasivo':
#     SUPORT_VEN_1 = 1
# if SUPORT_VEN == 'Sim, não invasivo':
#     SUPORT_VEN_2 = 1
# else:
#     SUPORT_VEN_3 = 1

# RENAL = st.selectbox('Paciente possui doença renal crônica?', options=['Sim', 'Não'])
# RENAL_1, RENAL_2 = 0, 0
# if RENAL == 'Sim':
#     RENAL_1 = 1
# else:
#     RENAL_2 = 1

# HEPATICA = st.selectbox('Paciente possui doença hepática crônica?', options=['Sim', 'Não'])
# HEPATICA_1, HEPATICA_2 = 0, 0
# if HEPATICA == 'Sim':
#     HEPATICA_1 = 1
# else:
#     HEPATICA_2 = 1

# Idade = st.selectbox('Qual é a idade do paciente?', options=['10-20', '21-30', '31-40', '41-50'])
# Idade_1, Idade_2, Idade_3, Idade_4 = 0, 0, 0, 0
# if Idade == '10-20':
#     Idade_1 = 1
# if Idade == '21-30':
#     Idade_2 = 1
# if Idade == '31-40':
#     Idade_3 = 1
# else:
#     Idade_4 = 1

# IMUNODEPRE = st.selectbox('Paciente possui imunodepressão?', options=['Sim', 'Não'])
# IMUNODEPRE_1, IMUNODEPRE_2 = 0, 0
# if IMUNODEPRE == 'Sim':
#     IMUNODEPRE_1 = 1
# else:
#     IMUNODEPRE_2 = 1

# CS_ESCOL_N = st.selectbox('Qual é a escolaridade do paciente?', options=['Sem escolaridade/ Analfabeto', 'Fundamental 1º ciclo (1ª a 5ª série)', 'Fundamental 2º ciclo (6ª a 9ª série)', 'Médio (1º ao 3º ano)', 'Superior'])
# CS_ESCOL_N_0, CS_ESCOL_N_1, CS_ESCOL_N_2, CS_ESCOL_N_3, CS_ESCOL_N_4 = 0, 0, 0, 0, 0
# if CS_ESCOL_N == 'Sem escolaridade/ Analfabeto':
#     CS_ESCOL_N_0 = 1
# if CS_ESCOL_N == 'Fundamental 1º ciclo (1ª a 5ª série)':
#     CS_ESCOL_N_1 = 1
# if CS_ESCOL_N == 'Fundamental 2º ciclo (6ª a 9ª série)':
#     CS_ESCOL_N_2 = 1
# if CS_ESCOL_N == 'Médio (1º ao 3º ano)':
#     CS_ESCOL_N_3 = 1
# else:
#     CS_ESCOL_N_4 = 1

# NEUROLOGIC = st.selectbox('Paciente possui doença neurológica crônica?', options=['Sim', 'Não'])
# NEUROLOGIC_1, NEUROLOGIC_2 = 0, 0
# if NEUROLOGIC == 'Sim':
#     NEUROLOGIC_1 = 1
# else:
#     NEUROLOGIC_2 = 1

# PUERPERA = st.selectbox('Paciente é puérpera?', options=['Sim', 'Não'])
# PUERPERA_1, PUERPERA_2 = 0, 0
# if PUERPERA == 'Sim':
#     PUERPERA_1 = 1
# else:
#     PUERPERA_2 = 1

# DOR_ABD = st.selectbox('Paciente apresentou dor abdominal?', options=['Sim', 'Não'])
# DOR_ABD_1, DOR_ABD_2 = 0, 0
# if DOR_ABD == 'Sim':
#     DOR_ABD_1 = 1
# else:
#     DOR_ABD_2 = 1

# PERD_PALA = st.selectbox('Paciente apresentou perda do paladar?', options=['Sim', 'Não'])
# PERD_PALA_1, PERD_PALA_2 = 0, 0
# if PERD_PALA == 'Sim':
#     PERD_PALA_1 = 1
# else:
#     PERD_PALA_2 = 1

# PERD_OLFT = st.selectbox('Paciente apresentou perda do olfato?', options=['Sim', 'Não'])
# PERD_OLFT_1, PERD_OLFT_2 = 0, 0
# if PERD_OLFT == 'Sim':
#     PERD_OLFT_1 = 1
# else:
#     PERD_OLFT_2 = 1

# DIARREIA = st.selectbox('Paciente apresentou diarreia?', options=['Sim', 'Não'])
# DIARREIA_1, DIARREIA_2 = 0, 0
# if DIARREIA == 'Sim':
#     DIARREIA_1 = 1
# else:
#     DIARREIA_2 = 1

# VOMITO = st.selectbox('Paciente apresentou vômito?', options=['Sim', 'Não'])
# VOMITO_1, VOMITO_2 = 0, 0
# if VOMITO == 'Sim':
#     VOMITO_1 = 1
# else:
#     VOMITO_2 = 1

# ASMA = st.selectbox('Paciente possui asma?', options=['Sim', 'Não'])
# ASMA_1, ASMA_2 = 0, 0
# if ASMA == 'Sim':
#     ASMA_1 = 1
# else:
#     ASMA_2 = 1

# DIABETES = st.selectbox('Paciente possui diabetes?', options=['Sim', 'Não'])
# DIABETES_1, DIABETES_2 = 0, 0
# if DIABETES == 'Sim':
#     DIABETES_1 = 1
# else:
#     DIABETES_2 = 1

# TOSSE = st.selectbox('Paciente apresentou tosse?', options=['Sim', 'Não'])
# TOSSE_1, TOSSE_2 = 0, 0
# if TOSSE == 'Sim':
#     TOSSE_1 = 1
# else:
#     TOSSE_2 = 1

# CLASSI_FIN = st.selectbox('Qual é a classificação final da doença?', options=['SRAG por influenza', 'SRAG por outro vírus respiratório', 'SRAG por outro agente etiológico', 'SRAG não especificado', 'SRAG por COVID-19'])
# CLASSI_FIN_1, CLASSI_FIN_2, CLASSI_FIN_3, CLASSI_FIN_4, CLASSI_FIN_5 = 0, 0, 0, 0, 0
# if CLASSI_FIN == 'SRAG por influenza':
#     CLASSI_FIN_1 = 1
# if CLASSI_FIN == 'SRAG por outro vírus respiratório':
#     CLASSI_FIN_2 = 1
# if CLASSI_FIN == 'SRAG por outro agente etiológico':
#     CLASSI_FIN_3 = 1
# if CLASSI_FIN == 'SRAG não especificado':
#     CLASSI_FIN_4 = 1
# else:
#     CLASSI_FIN_5 = 1

# GARGANTA = st.selectbox('Paciente apresentou dor de garganta?', options=['Sim', 'Não'])
# GARGANTA_1, GARGANTA_2 = 0, 0
# if GARGANTA == 'Sim':
#     GARGANTA_1 = 1
# else:
#     GARGANTA_2 = 1

# OBESIDADE = st.selectbox('Paciente possui obesidade?', options=['Sim', 'Não'])
# OBESIDADE_1, OBESIDADE_2 = 0, 0
# if OBESIDADE == 'Sim':
#     OBESIDADE_1 = 1
# else:
#     OBESIDADE_2 = 1

# CARDIOPATI = st.selectbox('Paciente possui cardiopatia crônica?', options=['Sim', 'Não'])
# CARDIOPATI_1, CARDIOPATI_2 = 0, 0
# if CARDIOPATI == 'Sim':
#     CARDIOPATI_1 = 1
# else:
#     CARDIOPATI_2 = 1

# FEBRE = st.selectbox('Paciente apresentou febre?', options=['Sim', 'Não'])
# FEBRE_1, FEBRE_2 = 0, 0
# if FEBRE == 'Sim':
#     FEBRE_1 = 1
# else:
#     FEBRE_2 = 1

# FADIGA = st.selectbox('Paciente apresentou fadiga?', options=['Sim', 'Não'])
# FADIGA_1, FADIGA_2 = 0, 0
# if FADIGA == 'Sim':
#     FADIGA_1 = 1
# else:
#     FADIGA_2 = 1

# VACINA = st.selectbox('Paciente é vacinado contra influenza?', options=['Sim', 'Não'])
# VACINA_1, VACINA_2 = 0, 0
# if VACINA == 'Sim':
#     VACINA_1 = 1
# else:
#     VACINA_2 = 1

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

#data = pd.DataFrame([['Brown_ethnicity', 'Superior_Scolarity', 'Cough_No', 'Dispnea_No', 'No_Distress_Respiratory', 'No_O2_Sat_low_95', 'Diarrhea_Yes', 'Olfaction_Loss_Yes', 'IOT_Yes', 'IOT_No', 'UTI_Yes', 'UTI_No', 'Not_Specificate_Cause', 'Age_21_30_Years_old', 'Third_Trimester_pregnancy']], columns=['Brown_ethnicity', 'Superior_Scolarity', 'Cough_No', 'Dispnea_No', 'No_Distress_Respiratory', 'No_O2_Sat_low_95', 'Diarrhea_Yes', 'Olfaction_Loss_Yes', 'IOT_Yes', 'IOT_No', 'UTI_Yes', 'UTI_No', 'Not_Specificate_Cause', 'Age_21_30_Years_old', 'Third_Trimester_pregnancy'])

#st.dataframe(data=data)

# data = pd.DataFrame([[Primeiro_Trimestre_1, Primeiro_Trimestre_2, Segundo_Trimestre_1, Segundo_Trimestre_2, Terceiro_Trimestre_1, Terceiro_Trimestre_2, UTI_1, UTI_2, DESC_RESP_1, DESC_RESP_2, DISPNEIA_1, DISPNEIA_2, SATURACAO_1, SATURACAO_2, HEMATOLOGI_1, HEMATOLOGI_2, CS_RACA_1, CS_RACA_2, CS_RACA_3, CS_RACA_4, CS_RACA_5, PNEUMOPATI_1, PNEUMOPATI_2,
# ANTIVIRAL_1, ANTIVIRAL_2, TOMO_RES_1, TOMO_RES_2, TOMO_RES_3, TOMO_RES_4, TOMO_RES_5, TOMO_RES_6, RAIOX_RES_1, RAIOX_RES_2, RAIOX_RES_3, RAIOX_RES_4, RAIOX_RES_5, RAIOX_RES_6, SUPORT_VEN_1, SUPORT_VEN_2, SUPORT_VEN_3, RENAL_1, RENAL_2,
# HEPATICA_1, HEPATICA_2, Idade_1, Idade_2, Idade_3, Idade_4, IMUNODEPRE_1, IMUNODEPRE_2, CS_ESCOL_N_0, CS_ESCOL_N_1, CS_ESCOL_N_2, CS_ESCOL_N_3, CS_ESCOL_N_4, NEUROLOGIC_1, NEUROLOGIC_2, PUERPERA_1, PUERPERA_2,
# DOR_ABD_1, DOR_ABD_2, PERD_PALA_1, PERD_PALA_2, PERD_OLFT_1, PERD_OLFT_2, DIARREIA_1, DIARREIA_2, VOMITO_1, VOMITO_2, ASMA_1, ASMA_2, DIABETES_1, DIABETES_2, TOSSE_1, TOSSE_2, CLASSI_FIN_1, CLASSI_FIN_2,
# CLASSI_FIN_3, CLASSI_FIN_4, CLASSI_FIN_5, GARGANTA_1, GARGANTA_2, OBESIDADE_1, OBESIDADE_2, CARDIOPATI_1, CARDIOPATI_2, FEBRE_1, FEBRE_2, FADIGA_1, FADIGA_2, VACINA_1, VACINA_2]], columns=['Primeiro_Trimestre_1', 'Primeiro_Trimestre_2', 'Segundo_Trimestre_1', 'Segundo_Trimestre_2', 'Terceiro_Trimestre_1', 'Terceiro_Trimestre_2', 'UTI_1', 'UTI_2', 'DESC_RESP_1', 'DESC_RESP_2', 'DISPNEIA_1', 'DISPNEIA_2', 'SATURACAO_1', 'SATURACAO_2', 'HEMATOLOGI_1', 'HEMATOLOGI_2', 'CS_RACA_1', 'CS_RACA_2', 'CS_RACA_3', 'CS_RACA_4', 'CS_RACA_5', 'PNEUMOPATI_1', 'PNEUMOPATI_2',
# 'ANTIVIRAL_1', 'ANTIVIRAL_2', 'TOMO_RES_1', 'TOMO_RES_2', 'TOMO_RES_3', 'TOMO_RES_4', 'TOMO_RES_5', 'TOMO_RES_6', 'RAIOX_RES_1', 'RAIOX_RES_2', 'RAIOX_RES_3', 'RAIOX_RES_4', 'RAIOX_RES_5', 'RAIOX_RES_6', 'SUPORT_VEN_1', 'SUPORT_VEN_2', 'SUPORT_VEN_3', 'RENAL_1', 'RENAL_2',
# 'HEPATICA_1', 'HEPATICA_2', 'Idade_1', 'Idade_2', 'Idade_3', 'Idade_4', 'IMUNODEPRE_1', 'IMUNODEPRE_2', 'CS_ESCOL_N_0', 'CS_ESCOL_N_1', 'CS_ESCOL_N_2', 'CS_ESCOL_N_3', 'CS_ESCOL_N_4', 'NEUROLOGIC_1', 'NEUROLOGIC_2', 'PUERPERA_1', 'PUERPERA_2',
# 'DOR_ABD_1', 'DOR_ABD_2', 'PERD_PALA_1', 'PERD_PALA_2', 'PERD_OLFT_1', 'PERD_OLFT_2', 'DIARREIA_1', 'DIARREIA_2', 'VOMITO_1', 'VOMITO_2', 'ASMA_1', 'ASMA_2', 'DIABETES_1', 'DIABETES_2', 'TOSSE_1', 'TOSSE_2', 'CLASSI_FIN_1', 'CLASSI_FIN_2',
# 'CLASSI_FIN_3', 'CLASSI_FIN_4', 'CLASSI_FIN_5', 'GARGANTA_1', 'GARGANTA_2', 'OBESIDADE_1','OBESIDADE_2', 'CARDIOPATI_1', 'CARDIOPATI_2', 'FEBRE_1', 'FEBRE_2', 'FADIGA_1', 'FADIGA_2', 'VACINA_1', 'VACINA_2'])

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