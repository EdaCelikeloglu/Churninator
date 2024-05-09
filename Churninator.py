import plotly.express as px
import streamlit as st
import time
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
from math import pi

from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from main import process_data
import warnings

warnings.simplefilter(action="ignore")

st.set_page_config(page_title="Churninator", page_icon="🤖", layout="wide")
alt.themes.enable("dark")


st.markdown("# Churninator")

theme_settings = {
    'color_discrete_sequence': px.colors.qualitative.Set1,  # Belirli bir renk paleti
    'template': 'plotly_white',  # Arka plan ve ızgara ayarları için şablon
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',  # Grafik arka planı şeffaf siyah
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',  # Kağıt arka planı şeffaf siyah
}

with st.sidebar:
    st.title('🤖 Churninator')


# st.markdown("# Analiz")
# st.sidebar.header("Analiz")


@st.cache_data
def load_data():
    df = pd.read_csv("BankChurners.csv")
    return df


# Veri setini yükleme
df = load_data()

import streamlit as st

import streamlit as st

# Add CSS to style the background and button
st.markdown(
    """
    <style>
    body {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        background-image: url('Background_image.svg');
        background-size: cover;
    }
    .button {
        background-color: purple;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        border: none;
        cursor: pointer;
    }
    .button:hover {
        background-color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add button
if st.button("Müşteri Bilgilerini Sunucudan Çekmek / Güncellemek İçin Tıklayın"):
    st.empty()  # Remove the button after being clicked


# st.markdown(
#     """
#     <style>
#     .button {
#         background-color: purple !important;
#     }
#     .button:hover {
#         background-color: blue !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
# result = st.button("Müşterileri Bilgilerini Sunucudan Çekmek İçin Tıklayın")
# if result:
#   st.write("Kaybetme riski ile karşı karşıya olduğunuz müşterilerinizi on farklı gruba ayırdık.")
#   st.write(formatted_string)
#   st.write("**Önerilerimiz:**")
#   st.write()

# def set_transparent_background(plt_object):
#     if isinstance(plt_object, plt.Axes) or isinstance(plt_object, plt.Figure):
#         # For Matplotlib and Seaborn plots
#         plt_object.patch.set_alpha(0)
#         plt_object.tick_params(axis='x', colors='white')
#         plt_object.tick_params(axis='y', colors='white')
#         plt_object.xaxis.label.set_color('white')
#         plt_object.yaxis.label.set_color('white')
#     elif hasattr(plt_object, 'update_layout'):
#         # For Plotly plots
#         plt_object.update_layout(plot_bgcolor='rgba(0,0,0,0)')
#
#     else:
#         st.error("Unsupported plot type")

def set_transparent_background(plt_object):
    if isinstance(plt_object, plt.Axes) or isinstance(plt_object, plt.Figure):
        # For Matplotlib and Seaborn plots
        plt_object.patch.set_alpha(0)
    elif hasattr(plt_object, 'update_layout'):
        # For Plotly plots
        plt_object.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        plt_object.update_layout(xaxis=dict(tickfont=dict(color='white')),
                                 yaxis=dict(tickfont=dict(color='white')),
                                 xaxis_title=dict(font=dict(color='white')),
                                 yaxis_title=dict(font=dict(color='white')))
    else:
        st.error("Unsupported plot type")


def grab_col_names(dataframe, cat_th=9, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_car: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


def combine_categories(df, cat_col1, cat_col2, new_col_name):
    df[new_col_name] = df[cat_col1].astype(str) + '_' + df[cat_col2].astype(str)


df.drop([
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"],
    inplace=True, axis=1)

# Bağımlı değişkenimizin ismini target yapalım ve 1, 0 atayalım:
df.rename(columns={"Attrition_Flag": "Target"}, inplace=True)
df["Target"] = df.apply(lambda x: 0 if (x["Target"] == "Existing Customer") else 1, axis=1)

# ID kolonunda duplicate bakıp, sonra bu değişkeni silme
df["CLIENTNUM"].nunique()  # 10127 - yani duplicate yok id'de
df.drop("CLIENTNUM", axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Outlier temizleme
# IQR
def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0])
    # if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
    #     print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    # else:
    #     print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


for col in num_cols:
    print(col, grab_outliers(df, col))


def remove_outliers_from_all_columns(dataframe):
    for col_name in num_cols:
        low, up = outlier_thresholds(dataframe, col_name)  # Aykırı değer sınırlarını hesapla
        outliers = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]
        print(f"{col_name} için aykırı değer sayısı: {outliers.shape[0]}")
        # Aykırı değerleri dataframe'den çıkar
        dataframe = dataframe.drop(outliers.index).reset_index(drop=True)
    return dataframe


df = remove_outliers_from_all_columns(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LOF
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols])

df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 100], style='.-')
plt.show()

th = np.sort(df_scores)[27]

df.drop(axis=0, labels=df[df_scores < th].index, inplace=True)  # Dropping the outliers.
df.head()

df = df.reset_index(drop=True)
# df.shape (10007, 20)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Missing values
cols_with_unknown = ['Income_Category', "Education_Level"]
for col in cols_with_unknown:
    df[col] = df[col].apply(lambda x: np.nan if x == 'Unknown' else x)

# ANALİZ BAŞLANGICI
df["On_book_cat"] = np.where((df["Months_on_book"] < 12), "<1_year", np.where((df["Months_on_book"] < 24), "<2_years",
                                                                              np.where((df["Months_on_book"] < 36),
                                                                                       "<3_years", np.where(
                                                                                      (df["Months_on_book"] < 48),
                                                                                      "<4_years", "<5_years"))))

df['Total_Amt_Increased'] = np.where((df['Total_Amt_Chng_Q4_Q1'] > 0) & (df['Total_Amt_Chng_Q4_Q1'] < 1), 0, 1)

df["Has_debt"] = np.where((df["Credit_Limit"] > df["Avg_Open_To_Buy"]), 1, 0).astype(int)

df["Important_client_score"] = df["Total_Relationship_Count"] * (df["Months_on_book"] / 12)

df["Avg_Trans_Amt"] = df["Total_Trans_Amt"] / df['Total_Trans_Ct']

labels = ['Young', 'Middle_Aged', 'Senior']
bins = [25, 35, 55, 74]
df['Customer_Age_Category'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels)

df["Days_Inactive_Last_Year"] = df["Months_Inactive_12_mon"] * 30

df["Days_Inactive_Last_Year"].replace(0, 30, inplace=True)
df["Days_Inactive_Last_Year"].replace(180, 150, inplace=True)

df["RecencyScore"] = df["Days_Inactive_Last_Year"].apply(lambda x: 5 if x == 30 else
4 if x == 60 else
3 if x == 90 else
2 if x == 120 else
1 if x == 150 else x)

df["MonetaryScore"] = pd.qcut(df["Total_Trans_Amt"], 5, labels=[1, 2, 3, 4, 5])
df["FrequencyScore"] = pd.qcut(df["Total_Trans_Ct"], 5, labels=[1, 2, 3, 4, 5])

combine_categories(df, 'Customer_Age_Category', 'Marital_Status', 'Age_&_Marital')
combine_categories(df, 'Gender', 'Customer_Age_Category', 'Gender_&_Age')
combine_categories(df, "Card_Category", "Customer_Age_Category", "Card_&_Age")
combine_categories(df, "Gender", "FrequencyScore", "Gender_&_Frequency")
combine_categories(df, "Gender", "MonetaryScore", "Gender_&_Monetary")

df['Total_Amt_Increased'] = np.where((df['Total_Amt_Chng_Q4_Q1'] >= 0) & (df['Total_Amt_Chng_Q4_Q1'] < 1), 0, 1)
df['Total_Ct_Increased'] = np.where((df['Total_Ct_Chng_Q4_Q1'] >= 0) & (df['Total_Ct_Chng_Q4_Q1'] < 1), 0, 1)

df.loc[(df["Total_Ct_Chng_Q4_Q1"] == 1) & (df["Total_Amt_Chng_Q4_Q1"] > 1), "Ct_vs_Amt"] = "Same_ct_inc_amt"
# boş
df.loc[(df["Total_Ct_Chng_Q4_Q1"] == 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Same_ct_same_amt"  # BOŞ
# İşlem sayısı aynı kalıp, harcama miktarı azalanlar: (harcamalardan mı kısıyorlar? belki ihtiyaçları olanları almışlardır.) TODO May_Marry ile incele)
df.loc[(df["Total_Ct_Chng_Q4_Q1"] == 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Same_ct_dec_amt"
# işlem sayısı da, miktarı da artmış (bizi sevindiren müşteri <3 )
df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] > 1), "Ct_vs_Amt"] = "Inc_ct_inc_amt"
# BOŞ İşlem sayısı artmasına rağmen, harcama miktarı aynı kalanlar: (aylık ortalama harcama azalıyor)
df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Inc_ct_same_amt"  # BOŞ
# İşlem sayısı artmış ama miktar azalmış. Yani daha sık, ama daha küçük alışverişler yapıyor. Bunlar düşük income grubuna aitse bankayı mutlu edecek bir davranış.
df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Inc_ct_dec_amt"
# (df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1)]).groupby("Income_Category").count() # Evet, düşük income grubuna ait.
# İşlem sayısı azalmış ama daha büyük miktarlarda harcama yapılıyor:
df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] > 1), "Ct_vs_Amt"] = "Dec_ct_inc_amt"
# İşlem sayısı azalmış, toplam miktar aynı kalmış (yani ortalama harcama artmış):
df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Dec_ct_same_amt"
df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Dec_ct_dec_amt"

# Personalar
df["Affluent_criteria"] = (df['Income_Category'] == '$120K +').astype("Int64")
df["Budget_criteria"] = ((df['Income_Category'] == 'Less than $40K') & (
    df['Education_Level'].isin(['High School', 'College']))).astype("Int64")
df["Young_prof_criteria"] = ((df['Customer_Age'] <= 30) & (df['Education_Level'].isin(['College', 'Graduate']))).astype(
    "Int64")
df["Family_criteria"] = (
            (df["Marital_Status"].isin(["Married", "Divorced", "Unknown"])) & (df['Dependent_count'] >= 3)).astype(int)
df["Credit_builder_criteria"] = (df['Credit_Limit'] < 2500).astype(
    int)  # This threshold is chosen to include individuals with credit limits in the lower percentiles of the distribution, which may indicate a need for credit-building strategies or entry-level credit products.
df["Digital_criteria"] = (df['Contacts_Count_12_mon'] == 0).astype(int)
df["High_net_worth_individual"] = ((df['Income_Category'] == '$120K +') & (df['Total_Trans_Amt'] > 5000)).astype(
    "Int64")
df["Rewards_maximizer"] = ((df['Total_Trans_Amt'] > 10000) & (df['Total_Revolving_Bal'] == 0)).astype(
    int)  # For the Rewards_maximizer column, the threshold for Total_Trans_Amt is also set at $10000. Since rewards maximizers are individuals who strategically maximize rewards and benefits from credit card usage, it's reasonable to expect that they engage in higher levels of spending. Therefore, the threshold of $10000 for Total_Trans_Amt appears appropriate for identifying rewards maximizers, considering that it captures individuals with relatively high spending habits.
df["May_marry"] = ((df["Age_&_Marital"] == "Young_Single") & (df['Dependent_count'] == 0)).astype(int)

df["Product_by_Year"] = df["Total_Relationship_Count"] / (df["Months_on_book"] / 12)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Rare analyser:
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "Target", cat_cols)

# Rare encoding:
df["Card_Category"] = df["Card_Category"].apply(lambda x: "Gold_Platinum" if x == "Platinum" or x == "Gold" else x)
df["Months_Inactive_12_mon"] = df["Months_Inactive_12_mon"].apply(lambda x: 1 if x == 0 else x)
df["Ct_vs_Amt"] = df["Ct_vs_Amt"].apply(lambda x: "Dec_ct_inc_amt" if x == "Dec_ct_same_amt" else x)
df["Ct_vs_Amt"] = df["Ct_vs_Amt"].apply(lambda x: "Inc_ct_inc_amt" if x == "Same_ct_inc_amt" else x)
df["Contacts_Count_12_mon"] = df["Contacts_Count_12_mon"].apply(lambda x: 5 if x == 6 else x)
df["Card_&_Age"] = df["Card_&_Age"].apply(lambda x: "Rare" if df["Card_&_Age"].value_counts()[x] < 30 else x)
df["Card_&_Age"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

categories_dict = {
    "Education_Level": ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', np.nan],
    "Income_Category": ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', np.nan],
    "Customer_Age_Category": ['Young', 'Middle_Aged', 'Senior'],
    "Card_Category": ['Blue', 'Silver', 'Gold_Platinum'],
    "On_book_cat": ["<2_years", "<3_years", "<4_years", "<5_years"]}


def ordinal_encoder(dataframe, col):
    if col in categories_dict:
        col_cats = categories_dict[col]
        ordinal_encoder = OrdinalEncoder(categories=[col_cats])
        dataframe[col] = ordinal_encoder.fit_transform(dataframe[[col]])

    return dataframe


for col in df.columns:
    ordinal_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df = one_hot_encoder(df, ["Gender"], drop_first=True)  # M'ler 1.
df.rename(columns={"Gender_M": "Gender"}, inplace=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.isnull().sum()

# knn eski
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = [col for col in df.columns if col not in numeric_columns]
df_numeric = df[numeric_columns]
imputer = KNNImputer(n_neighbors=10)
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_columns)
df = pd.concat([df_numeric_imputed, df[categorical_columns]], axis=1)
df["Education_Level"] = df["Education_Level"].round().astype("Int64")
df["Income_Category"] = df["Income_Category"].round().astype("Int64")

cat_cols, num_cols, cat_but_car = grab_col_names(df)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
df.drop(useless_cols, axis=1, inplace=True)

df.head()
dff = df.copy()
df = dff.copy()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Değişken tipi dönüştürme:
for col in df.columns:
    if df[col].dtype == 'float64':  # Sadece float sütunları kontrol edelim
        if (df[col] % 1 == 000).all():  # Tüm değerlerin virgülden sonrası 0 mı kontrol edelim
            df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# rfm skorları ile segmentasyon oluşturma
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Lose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'}

# segment oluşturma (Recency + Frequency)
df['Segment'] = df['RecencyScore'].astype(str) + df['FrequencyScore'].astype(str)
df['Segment'] = df['Segment'].replace(seg_map, regex=True)
df.head(40)

# 33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
# tüm görselleştirme

#####
# def calculate_churn(dataframe, input_year):
#   selected_year_data = dataframe[dataframe['year'] == input_year].reset_index()
#   previous_year_data = dataframe[dataframe['year'] == input_year - 1].reset_index()
#   selected_year_data['population_difference'] = selected_year_data.population.sub(previous_year_data.population, fill_value=0)
#   return pd.concat([selected_year_data.states, selected_year_data.id, selected_year_data.population, selected_year_data.population_difference], axis=1).sort_values(by="population_difference", ascending=False)
#
#
# def make_donut(input_response, input_text, input_color):
#     input_response = dataframe["Target"]==1.sum()/len
#     if input_color == 'blue':
#         chart_color = ['#29b5e8', '#155F7A']
#     if input_color == 'green':
#         chart_color = ['#27AE60', '#12783D']
#     if input_color == 'orange':
#         chart_color = ['#F39C12', '#875A12']
#     if input_color == 'red':
#         chart_color = ['#E74C3C', '#781F16']
#
#
#     source = pd.DataFrame({
#         "Topic": ['', input_text],
#         "% value": [100 - input_response, input_response]
#     })
#     source_bg = pd.DataFrame({
#         "Topic": ['', input_text],
#         "% value": [100, 0]
#     })
#
#     plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
#         theta="% value",
#         color=alt.Color("Topic:N",
#                         scale=alt.Scale(
#                             # domain=['A', 'B'],
#                             domain=[input_text, ''],
#                             # range=['#29b5e8', '#155F7A']),  # 31333F
#                             range=chart_color),
#                         legend=None),
#     ).properties(width=130, height=130)
#
#     text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700,
#                           fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
#     plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
#         theta="% value",
#         color=alt.Color("Topic:N",
#                         scale=alt.Scale(
#                             # domain=['A', 'B'],
#                             domain=[input_text, ''],
#                             range=chart_color),  # 31333F
#                         legend=None),
#     ).properties(width=130, height=130)
#     return plot_bg + plot + text


##########


###### Dashboard kolonları başlangıç

col = st.columns([0.5, 0.5], gap='small')

with col[0]:
    st.markdown('#### Gains/Losses')

    # PCA hazırlık
    df1 = df.copy()
    df1 = one_hot_encoder(df1, ["Marital_Status",
                                "Age_&_Marital",
                                "Gender_&_Age",
                                "Card_&_Age",
                                "Gender_&_Frequency",
                                "Gender_&_Monetary",
                                'Ct_vs_Amt',
                                'Dependent_count',
                                'Total_Relationship_Count',
                                'Months_Inactive_12_mon',
                                'Contacts_Count_12_mon'],
                          drop_first=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(df1)

    scal_cols = ['Customer_Age',
                 'Months_on_book',
                 'Credit_Limit',
                 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy',
                 'Total_Trans_Amt',
                 'Total_Trans_Ct',
                 'Important_client_score',
                 'Avg_Trans_Amt']

    rs = RobustScaler()
    df1[scal_cols] = rs.fit_transform(df1[scal_cols])

    features_scaled = df1.drop(['Segment'], axis=1)
    # PCA uygulama (3 ana bileşen)
    pca = PCA(n_components=3)
    components = pca.fit_transform(features_scaled)

    # PCA sonuçlarını DataFrame'e dönüştürme
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Segment'] = df1['Segment']  # Renklendirme için Segment sütununu ekleme

    # 3D scatter plot oluşturma
    fig_pca = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Segment',
                            title="PCA Results (3D) Colored by Segment")
    fig_pca.update_traces(marker=dict(size=3))
    fig_pca.update_layout(width=1000, height=800)
    st.plotly_chart(fig_pca)

    ####
    # Ürün sayısı arttıkça Churn olasılığı azalıyor
    st.write("Ürün sayısı arttıkça Churn olasılığı azalıyor.")
    mean_target_by_relationship = df.groupby("Total_Relationship_Count")["Target"].mean().reset_index()
    fig = px.bar(mean_target_by_relationship, x="Total_Relationship_Count", y="Target",
                 labels={"Total_Relationship_Count": "Toplam Ürün Sayısı", "Target": "Target"},
                 title="Toplam Ürün Sayısına Göre Target", color_discrete_sequence=["blue"])
    st.plotly_chart(fig)

    # Target'e göre Important_client_score'un grafiği:
    mean_scores_by_target = df.groupby("Target")["Important_client_score"].mean().reset_index()
    fig = px.bar(mean_scores_by_target, x="Target", y="Important_client_score",
                 labels={"Target": "Hedef", "Important_client_score": "Ortalama Puan"},
                 title="Target'a Göre Important Client Score")
    fig.update_layout(height=400, width=400)
    st.plotly_chart(fig)

    # Borcu çok olanlar gidemiyor
    mean_utilization_by_target = df.groupby("Target")["Avg_Utilization_Ratio"].mean().reset_index()
    mean_revolving_bal_by_target = df.groupby("Target")["Total_Revolving_Bal"].mean().reset_index()
    fig_utilization = px.bar(mean_utilization_by_target, x="Target", y="Avg_Utilization_Ratio",
                             labels={"Target": "Hedef", "Avg_Utilization_Ratio": "Borç/Kredi Limiti"},
                             title="Target'a Göre Borç/Kredi Limiti",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_utilization.update_layout(height=400, width=400)
    fig_revolving_bal = px.bar(mean_revolving_bal_by_target, x="Target", y="Total_Revolving_Bal",
                               labels={"Target": "Hedef", "Total_Revolving_Bal": "Ortalama Devir Bakiyesi"},
                               title="Hedefe Göre Ortalama Devir Bakiyesi",
                               color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_revolving_bal.update_layout(height=400, width=400)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_utilization)
    with col2:
        st.plotly_chart(fig_revolving_bal)

with col[1]:
    st.markdown('#### Total Population')

    # Heatmap
    data = {
        '1': [0.000, 0.515, 0.552, 0.532, 0.616, 0.000],
        '2': [0.818, 0.594, 0.575, 0.602, 0.610, 0.528],
        '3': [0.636, 0.750, 0.939, 0.895, 0.979, 0.711],
        '4': [0.800, 1.198, 1.024, 1.137, 1.071, 1.171],
        '5': [1.403, 1.442, 1.274, 1.390, 1.308, 1.699],
        '6': [2.052, 2.057, 2.033, 2.043, 2.034, 1.951],
        '7': [2.974, 2.979, 2.842, 3.006, 2.924, 3.117]}

    index = [0, 250, 750, 1250, 1750, 2250]

    grouped_data = pd.DataFrame(data, index=index)

    # Plot the heatmap-like graph
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = sns.heatmap(grouped_data, cmap='plasma', annot=True, fmt=".2f", ax=ax, vmin=0, vmax=4)
    ax.invert_yaxis()

    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Customize the legend labels
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks(range(5))
    colorbar.set_ticklabels(['0: Less than $40K', '1: $40K - $60K', '2: $60K - $80K', '3: $80K - $120K', '4: $120K +'])

    for i in range(len(grouped_data)):
        for j in range(len(grouped_data.columns)):
            value = grouped_data.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f'{value:.2f}', ha='center', va='center', color='white')

    # Customize x-axis and y-axis labels and ticks
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])  # Setting the positions of the ticks
    ax.set_xticklabels(['0', '5k', '10k', '15k', '20k', '25k', '30k', '35k'])
    ax.set_xlabel('Credit Limit (USD)')

    ax.set_yticks([0, 1, 2, 3, 4, 5])  # Setting the positions of the ticks
    ax.set_yticklabels(['0', '500', '1000', '1500', '2000', '2500'])
    ax.set_ylabel('Total Revolving Balance (USD)')

    set_transparent_background(fig)
    st.pyplot(fig)

    # Treemap
    fig = px.treemap(df, path=['Target', 'Segment'], title="Target ve Segment")
    fig.update_layout(height=600, width=800)
    st.plotly_chart(fig)

    # Gelir kategorilerine göre ortalama devir bakiyesi
    fig = px.bar(df, x="Income_Category", y="Total_Revolving_Bal",
                 labels={"Income_Category": "Gelir Kategorisi", "Total_Revolving_Bal": "Ortalama Devir Bakiyesi"},
                 title="Gelir Kategorisine Göre Ortalama Devir Bakiyesi",
                 color="Income_Category", color_discrete_sequence=px.colors.qualitative.Pastel)
    # Grafiği görüntüle
    st.plotly_chart(fig, use_container_width=True)

    # Age_&_Marital   Gender_&_Age        Card_&_Age değişkenlerini target ile baktım:
    # fig_age_marital = px.histogram(df, x="Age_&_Marital", color="Target", barmode="group",
    #                                labels={"Age_&_Marital": "Yaş ve Medeni Durum", "Target": "Churn Durumu"},
    #                                title="Yaş ve Medeni Duruma Göre Churn Durumu")
    # fig_age_marital.update_layout(xaxis_title="Yaş ve Medeni Durum", yaxis_title="Sayı")
    #
    # fig_gender_age = px.histogram(df, x="Gender_&_Age", color="Target", barmode="group",
    #                               labels={"Gender_&_Age": "Cinsiyet ve Yaş", "Target": "Churn Durumu"},
    #                               title="Cinsiyet ve Yaşa Göre Churn Durumu")
    # fig_gender_age.update_layout(xaxis_title="Cinsiyet ve Yaş", yaxis_title="Sayı")
    #
    # fig_card_age = px.histogram(df, x="Card_&_Age", color="Target", barmode="group",
    #                             labels={"Card_&_Age": "Kart ve Yaş", "Target": "Churn Durumu"},
    #                             title="Kart ve Yaşa Göre Churn Durumu")
    # fig_card_age.update_layout(xaxis_title="Kart ve Yaş", yaxis_title="Sayı")
    # st.plotly_chart(fig_age_marital)
    # st.plotly_chart(fig_gender_age)
    # st.plotly_chart(fig_card_age)

    df_grouped = df.groupby(["Age_&_Marital", "Gender_&_Age", "Card_&_Age"])["Target"].value_counts(
        normalize=True).rename("Ratio").reset_index()

    # Filter the DataFrame to include only rows where Target=1
    df_target1 = df_grouped[df_grouped["Target"] == 1]

    # Merge with the total count of each category
    df_merged = pd.merge(df_grouped, df_target1[["Age_&_Marital", "Gender_&_Age", "Card_&_Age", "Ratio"]],
                         on=["Age_&_Marital", "Gender_&_Age", "Card_&_Age"], suffixes=('', '_target1'))

    # Plot histograms using the calculated ratio
    fig_age_marital = px.histogram(df_merged, x="Age_&_Marital", color="Target", barmode="group",
                                   labels={"Age_&_Marital": "Yaş ve Medeni Durum", "Target": "Churn Durumu"},
                                   title="Yaş ve Medeni Duruma Göre Churn Durumu",
                                   histfunc="sum", y="Ratio",
                                   category_orders={"Age_&_Marital": df_merged["Age_&_Marital"].unique()},
                                   color_discrete_map={0: 'blue', 1: 'red'})

    fig_gender_age = px.histogram(df_merged, x="Gender_&_Age", color="Target", barmode="group",
                                  labels={"Gender_&_Age": "Cinsiyet ve Yaş", "Target": "Churn Durumu"},
                                  title="Cinsiyet ve Yaşa Göre Churn Durumu",
                                  histfunc="sum", y="Ratio",
                                  category_orders={"Gender_&_Age": df_merged["Gender_&_Age"].unique()},
                                  color_discrete_map={0: 'blue', 1: 'red'})

    fig_card_age = px.histogram(df_merged, x="Card_&_Age", color="Target", barmode="group",
                                labels={"Card_&_Age": "Kart ve Yaş", "Target": "Churn Durumu"},
                                title="Kart ve Yaşa Göre Churn Durumu",
                                histfunc="sum", y="Ratio",
                                category_orders={"Card_&_Age": df_merged["Card_&_Age"].unique()},
                                color_discrete_map={0: 'blue', 1: 'red'})

    # Update layout
    fig_age_marital.update_layout(xaxis_title="Yaş ve Medeni Durum", yaxis_title="Oran")
    fig_gender_age.update_layout(xaxis_title="Cinsiyet ve Yaş", yaxis_title="Oran")
    fig_card_age.update_layout(xaxis_title="Kart ve Yaş", yaxis_title="Oran")

    # Display the histograms
    st.plotly_chart(fig_age_marital)
    st.plotly_chart(fig_gender_age)
    st.plotly_chart(fig_card_age)

    # Filtered DataFrames
    filtered_df1 = df[df['Target'] == 1]
    filtered_df0 = df[df['Target'] == 0]

    # Kategorik değişkenler ve renkler
    categories = ['Gender', 'Contacts_Count_12_mon', 'Total_Relationship_Count', "Months_Inactive_12_mon",
                  'Marital_Status']
    colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:pink', 'tab:orange']

    # Figür oluştur, 2 subplot ile (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=1000, subplot_kw={'projection': 'polar'})

    # Target 1 için
    ax = axes[0]
    total_categories = sum(filtered_df1[cat].nunique() for cat in categories)
    angles = np.linspace(0, 2 * np.pi, total_categories, endpoint=False)
    bar_width = (2 * np.pi / total_categories) * 0.8  # %80 genişlik, %20 boşluk
    start = 0
    for i, category in enumerate(categories):
        unique_vals = df[category].unique()
        value_counts = filtered_df1[category].value_counts().reindex(unique_vals, fill_value=0)
        category_angles = angles[start:start + len(unique_vals)]
        bars = ax.bar(category_angles, value_counts, width=bar_width, color=colors[i], alpha=0.6, label=category,
                      bottom=600)
        # Kategori değerlerinin isimlerini her barın üstüne yazma
        for bar, label in zip(bars, value_counts.index):
            angle = bar.get_x() + bar_width / 2  # Metni barın merkezine yerleştir
            height = 800
            ax.text(angle, height, str(label), color='black', ha='left', va='center', rotation=np.degrees(angle),
                    rotation_mode='anchor', fontsize=7)
        start += len(unique_vals)
        ax.text(0, 0, "1", color='black', ha='center', va='center', fontsize=12)
    fig.legend()

    # Target 0 için
    ax = axes[1]
    total_categories = sum(filtered_df0[cat].nunique() for cat in categories)
    angles = np.linspace(0, 2 * np.pi, total_categories, endpoint=False)
    start = 0
    for i, category in enumerate(categories):
        unique_vals = df[category].unique()
        value_counts = filtered_df0[category].value_counts().reindex(unique_vals, fill_value=0)
        category_angles = angles[start:start + len(unique_vals)]
        bars = ax.bar(category_angles, value_counts, width=bar_width, color=colors[i], alpha=0.6, label=category,
                      bottom=3000)
        # Kategori değerlerinin isimlerini her barın üstüne yazma
        for bar, label in zip(bars, value_counts.index):
            angle = bar.get_x() + bar_width / 2  # Metni barın merkezine yerleştir
            height = 3800
            ax.text(angle, height, str(label), color='black', ha='left', va='center', rotation=np.degrees(angle),
                    rotation_mode='anchor', fontsize=7)
        start += len(unique_vals)
        ax.text(0, 0, "0", color='black', ha='center', va='center', fontsize=12)

    # Ortak ayarlar
    for ax in axes:
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['polar'].set_visible(False)

    # Streamlit'te göster
    set_transparent_background(fig)
    st.pyplot(fig)

    # Radar grafiği
    # ------- PART 0: Reverse MinMax Scaler
    df['FrequencyScore'] = df['FrequencyScore'].cat.codes
    df['MonetaryScore'] = df['MonetaryScore'].cat.codes

    # Set data
    df_radar = pd.DataFrame({
        'group': ["Staying Customer", 'Churned Customer'],
        'Relationship Count': [df[df["Target"] == 0]["Total_Relationship_Count"].mean(),
                               df[df["Target"] == 1]["Total_Relationship_Count"].mean()],
        'Recency Score': [df[df["Target"] == 0]["RecencyScore"].mean(), df[df["Target"] == 1]["RecencyScore"].mean()],
        'Frequency Score': [df[df["Target"] == 0]["FrequencyScore"].mean(),
                            df[df["Target"] == 1]["FrequencyScore"].mean()],
        'Monetary Score': [df[df["Target"] == 0]["MonetaryScore"].mean(),
                           df[df["Target"] == 1]["MonetaryScore"].mean()],
        '(6 - Contact Count)': [6 - (df[df["Target"] == 0]["Contacts_Count_12_mon"].mean()),
                                6 - (df[df["Target"] == 1]["Contacts_Count_12_mon"].mean())],
    })
    # todo burada düz contact_count değil 6-contact count aldım. Bu şekilde, "Grafikte çıkan şekilde hacim büyüdükçe churn azalıyor" diyebiliriz. Ama bunu bir konuşalım.

    # ------- PART 1: Create background
    # number of variable
    categories = list(df_radar)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    # ax = plt.subplot(111, polar=True)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3], ["1", "2", "3"], color="grey", size=7)
    plt.ylim(0, 4)

    # ------- PART 2: Add plots
    # Plot each individual = each line of the data

    # Ind1
    values = df_radar.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Staying Customer")
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values = df_radar.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label='Churned Customer')
    ax.fill(angles, values, 'g', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Show the graph
    set_transparent_background(fig)
    st.pyplot(fig)

    # personlar için grafik?
    # persona = ["May_marry", "Credit_builder_criteria", "Family_criteria"]
    #
    # grid = [st.rows(3) for _ in range(3)]
    # current_row = 0
    # col = 0
    #
    # # Her bir feature için Target yüzdesini hesaplama ve grafik oluşturma
    # for feature in persona:
    #     if feature in df.columns:
    #         # Feature 1 olan kayıtları filtrele
    #         filtered_df = df[df[feature] == 1]
    #
    #         # Target 1 olanların yüzdesini hesapla
    #         if not filtered_df.empty:
    #             percentage = (filtered_df['Target'].sum() / filtered_df.shape[0]) * 100
    #         else:
    #             percentage = 0  # Eğer feature 1 hiç yoksa
    #
    #         # Yarım daire grafik oluştur
    #         fig = go.Figure(go.Indicator(
    #             mode="gauge+number",
    #             value=percentage,
    #             domain={'x': [0, 1], 'y': [0, 1]},
    #             title={'text': f"{feature}", 'font': {'size': 16}},
    #             gauge={
    #                 'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
    #                 'bar': {'color': "green"},
    #                 'bgcolor': "white",
    #                 'borderwidth': 2,
    #                 'bordercolor': "gray",
    #                 'steps': [
    #                     {'range': [0, percentage], 'color': 'lavender'},
    #                     {'range': [percentage, 100], 'color': 'mintcream'}],
    #                 'threshold': {
    #                     'line': {'color': "red", 'width': 4},
    #                     'thickness': 0.75,
    #                     'value': percentage}}))
    #
    #         # Uygun sütunda Streamlit'te göster
    #         with grid[row][current_row]:
    #             st.plotly_chart(fig, use_container_width=True)
    #
    #         current_row += 1
    #         if current_row > 2:
    #             current_row = 0
    #             col += 1

    persona = ["May_marry", "Credit_builder_criteria", "Family_criteria"]

    grid = [st.columns(3) for _ in range(3)]
    current_column = 0
    row = 0

    # Her bir feature için Target yüzdesini hesaplama ve grafik oluşturma
    for feature in persona:
        if feature in df.columns:
            # Feature 1 olan kayıtları filtrele
            filtered_df = df[df[feature] == 1]

            # Target 1 olanların yüzdesini hesapla
            if not filtered_df.empty:
                percentage = (filtered_df['Target'].sum() / filtered_df.shape[0]) * 100
            else:
                percentage = 0  # Eğer feature 1 hiç yoksa

            # Yarım daire grafik oluştur
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{feature}", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "green"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, percentage], 'color': 'lavender'},
                        {'range': [percentage, 100], 'color': 'mintcream'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': percentage}}))

            fig.update_layout(
                width=50,  # Genişlik
                height=200,  # Yükseklik
                margin=dict(l=10, r=10, t=10, b=10),  # Kenar boşlukları
                showlegend=False  # Açıklama gösterme
            )

            # Streamlit'te grafikleri göster
            st.plotly_chart(fig, use_container_width=True)

            current_column += 1
            if current_column > 2:
                current_column = 0
                row += 1
    import plotly.graph_objects as go

# # Yeni gelen müşteriler risk mi?
# st.write("Yeni gelen müşteriler risk mi?")
# mean_target_by_inactive_months = df.groupby("Months_Inactive_12_mon")["Target"].mean().reset_index()
# fig = px.bar(mean_target_by_inactive_months, x="Months_Inactive_12_mon", y="Target",
#              labels={"Months_Inactive_12_mon": "İnaktif Ay Sayısı", "Target": "Target"},
#              title="İnaktif Ay Sayısına Göre Target", color_discrete_sequence=px.colors.qualitative.Pastel)
# st.plotly_chart(fig)


# Dilara says: bunu Radar grafiğinde verdiğimiz için tekrar ayrıca vermeyelim diye düşünüyorum.
# müşteri ile iletişim sayısı ve target:
# mean_churn_by_contact = df.groupby("Contacts_Count_12_mon")["Target"].mean().reset_index()
# mean_churn_by_contact = mean_churn_by_contact.rename(columns={"Target": "Churn_Rate"})
# fig = px.bar(mean_churn_by_contact, x="Contacts_Count_12_mon", y="Churn_Rate",
#              labels={"Contacts_Count_12_mon": "İletişim Sayısı", "Churn_Rate": "Ortalama Churn Oranı"},
#              title="İletişim Sayısına Göre Ortalama Churn Oranı")
# fig.update_layout(height=400, width=400)
# st.plotly_chart(fig)


# burada Dec_ct_dec_amt kategorisi nedir? Çok fazla yoğunluk var orda
# Dilara says: burası sanki çok bir şey söylemiyor ya
#  Ct_vs_Amt ile Target:
# fig = px.histogram(df, x="Ct_vs_Amt", color="Target", barmode="group",
#                    title="Ct_vs_Amt Değişkeninin Target İle İlişkisi",
#                    labels={"Ct_vs_Amt": "Ct_vs_Amt", "Target": "Target Ortalaması"},
#                    color_discrete_map={0: "lightblue", 1: "salmon"})
#
# fig.update_layout(bargap=0.1)
# st.plotly_chart(fig)

# bunun notunu almışım birlikte yorumlayalım.:
# fig = px.scatter(df, x="Credit_Limit", y="Total_Revolving_Bal", color="Income_Category",
#                  title="Kredi Limiti ve Devir Bakiyesi İlişkisi",
#                  labels={"Credit_limit": "Kredi Limiti", "Total_revolving_Bal": "Devir Bakiyesi"},
#                  color_discrete_sequence=px.colors.qualitative.Set2)
# fig.update_layout(height=800, width=1200)
# st.plotly_chart(fig)

# df["Credit_Limit_bin"] = pd.qcut(df["Credit_Limit"], 7, labels=[1, 2, 3, 4, 5, 6, 7])
# df["Total_Revolving_Bal_bin"] = np.where((df["Total_Revolving_Bal"] <= 500), 250, np.where((df["Total_Revolving_Bal"] <= 1000), 750, np.where((df["Total_Revolving_Bal"] <= 1500), 1250, np.where((df["Total_Revolving_Bal"] <= 2000), 1750, np.where((df["Total_Revolving_Bal"] <= 2500), 2250, 0)))))
# grouped_data = df.groupby(['Total_Revolving_Bal_bin', 'Credit_Limit_bin'])['Income_Category'].mean().unstack(


st.markdown("<center>© 2024  DEG Bilgi Teknolojileri Danışmanlık ve Dağıtım A.Ş. Tüm hakları saklıdır.</center>", unsafe_allow_html=True)