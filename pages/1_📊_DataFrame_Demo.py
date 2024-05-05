import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import plotly as px
import plotly.express as px
import plotly.graph_objects as go
from pywaffle import Waffle
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
                             classification_report, RocCurveDisplay)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from xgboost import XGBClassifier
import graphviz
import joblib
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import seaborn as sns
import warnings
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


st.set_page_config(page_title="DataFrame Demo", page_icon="📊", layout="wide")

st.markdown("# DataFrame Demo")
st.sidebar.header("DataFrame Demo")
st.write(
    """Burada Dataframemimizin bir tanıtımı yer alacak."""
)

st.write("Kaggle linkine buradan ulaşabilirsiniz: [Veri Seti](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)")


df = pd.read_csv("BankChurners.csv")

df.drop([
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"],
    inplace=True, axis=1)

st.write(df.head())











# Hedef değişken dağılımını pasta grafiği olarak gösterme
attrition_counts = df["Attrition_Flag"].value_counts()
colors = px.colors.qualitative.Pastel
fig_attrition = px.pie(names=attrition_counts.index, values=attrition_counts.values, labels={"Attrition_Flag": "Sayı"},
                       title="Target Dağılımı", color=attrition_counts.index, color_discrete_map={attrition: colors[i] for i, attrition in enumerate(attrition_counts.index)})
fig_attrition.update_layout(height=400, width=400)
fig_attrition.update_traces(marker=dict(line=dict(color='black', width=1)))  # Kenarlıkları siyah yapma

# Cinsiyet dağılımını çubuk grafikle gösterme
gender_counts = df["Gender"].value_counts()
fig_gender = px.bar(x=gender_counts.index, y=gender_counts.values, labels={"x": "Cinsiyet", "y": "Sayı"},
                    title="Cinsiyet Dağılımı", color=gender_counts.index, color_discrete_map={gender: colors[i] for i, gender in enumerate(gender_counts.index)})
fig_gender.update_layout(height=400, width=400)
fig_gender.update_traces(marker=dict(line=dict(width=1)))

# Grafikleri yan yana gösterme
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_attrition)

with col2:
    st.plotly_chart(fig_gender)

# Gelir kategorilerine göre sayıları hesapla
income_counts = df["Income_Category"].value_counts()
fig_income = px.bar(x=income_counts.index, y=income_counts.values, labels={"x": "Gelir Kategorisi", "y": "Sayı"},
                    title="Gelir Kategorisi Dağılımı", color=income_counts.index,
                    color_discrete_map={income: colors[i] for i, income in enumerate(income_counts.index)})
fig_income.update_layout(height=400, width=400, showlegend=False)  # Grafik boyutunu ayarla

# Total_Revolving_Bal
fig_revolving_bal_hist = px.histogram(df, x="Total_Revolving_Bal", title="Toplam Devir Bakiyesi")
fig_revolving_bal_hist.update_layout(height=400, width=400)  # Grafik boyutunu ayarla

# Grafikleri yan yana gösterme
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_income)
with col2:
    st.plotly_chart(fig_revolving_bal_hist)



# Ürün sayısı
total_relationship_counts = df["Total_Relationship_Count"].value_counts()
fig_total_relationship = go.Figure(go.Bar(x=total_relationship_counts.index, y=total_relationship_counts.values,
                                          marker=dict(color=px.colors.qualitative.Set2)))

fig_total_relationship.update_layout(title="Müşterilerin Ürün Sayısı",
                                     xaxis_title="Ürün Sayısı",
                                     yaxis_title="Sayı", height=400, width=400)

# Months_Inactive_12_mon
months_inactive_counts = df["Months_Inactive_12_mon"].value_counts()
fig_inactive_months = go.Figure(go.Bar(x=months_inactive_counts.index, y=months_inactive_counts.values,
                                       marker=dict(color=px.colors.qualitative.Set2)))

fig_inactive_months.update_layout(title="Müşterilerin Son Bir Yılda İnaktif Geçirdiği Ay Sayısı",
                                  xaxis_title="Inaktif Ay Sayısı",
                                  yaxis_title="Sayı", height=400, width=400)

# Grafikleri yan yana gösterme
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_total_relationship)
with col2:
    st.plotly_chart(fig_inactive_months)




# Months_on_book
fig = px.histogram(df, x="Months_on_book", title="Müşterilerin Bankada Geçirdiği Ay Sayısı Dağılımı")
st.plotly_chart(fig)



