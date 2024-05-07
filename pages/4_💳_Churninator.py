import streamlit as st
import plotly.express as px
import streamlit as st
import time
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.simplefilter(action="ignore")

st.set_page_config(page_title="Model Demo", page_icon="💳", layout="wide")


st.markdown("# Churninator")
st.sidebar.header("Churninator")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)


@st.cache_data
def load_and_process_data():
    # CSV dosyasını yükle
    df = pd.read_csv("BankChurners.csv")

    # İstenmeyen sütunları düşür
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

    return df

# Veriyi önbelleğe al
df = load_and_process_data()


# Fonksiyonları tanımlayın ve cache'leyin
@st.cache_data
def grab_col_names(dataframe, cat_th=9, car_th=20):
    #cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car


@st.cache_data
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

@st.cache_data
def combine_categories(df, cat_col1, cat_col2, new_col_name):
    df[new_col_name] = df[cat_col1].astype(str) + '_' + df[cat_col2].astype(str)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

st.write(df.shape)
@st.cache_data
def detect_outliers(df, num_cols):
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(df[num_cols])
    df_scores = clf.negative_outlier_factor_
    th = np.sort(df_scores)[25]
    return df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# Outlierları bulma işlemi için önbelleğe alınmış sonucu çağırın
outliers_df = detect_outliers(df, num_cols)

st.write(df.shape)

@st.cache_data
def preprocess_data(df):
    cols_with_unknown = ['Income_Category', "Education_Level"]
    for col in cols_with_unknown:
        df[col] = df[col].apply(lambda x: np.nan if x == 'Unknown' else x)

    df["On_book_cat"] = np.where((df["Months_on_book"] < 12), "<1_year", np.where((df["Months_on_book"] < 24), "<2_years", np.where((df["Months_on_book"] < 36), "<3_years", np.where((df["Months_on_book"] < 48), "<4_years", "<5_years"))))
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

    return df

# Veri ön işleme işlemi için önbelleğe alınmış sonucu çağırın
df = preprocess_data(df)

st.write(df.head())

# import streamlit as st
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans
# from yellowbrick.cluster import KElbowVisualizer
#
# @st.cache_data
# def calculate_RFM(df):
#     # RFM
#     df["RecencyScore"] = df["Days_Inactive_Last_Year"].apply(lambda x: 5 if x == 30 else
#                                                             4 if x == 60 else
#                                                             3 if x == 90 else
#                                                             2 if x == 120 else
#                                                             1 if x == 150 else x)
#
#     df["MonetaryScore"] = pd.qcut(df["Total_Trans_Amt"], 5, labels=[1, 2, 3, 4, 5])
#     df["FrequencyScore"] = pd.qcut(df["Total_Trans_Ct"], 5, labels=[1, 2, 3, 4, 5])
#
#     seg_map = {
#             r'[1-2][1-2]': 'Hibernating',
#             r'[1-2][3-4]': 'At Risk',
#             r'[1-2]5': 'Can\'t Lose',
#             r'3[1-2]': 'About to Sleep',
#             r'33': 'Need Attention',
#             r'[3-4][4-5]': 'Loyal Customers',
#             r'41': 'Promising',
#             r'51': 'New Customers',
#             r'[4-5][2-3]': 'Potential Loyalists',
#             r'5[4-5]': 'Champions'
#     }
#
#     # segment oluşturma (Recency + Frequency)
#     df['Segment'] = df['RecencyScore'].astype(str) + df['FrequencyScore'].astype(str)
#     df['Segment'] = df['Segment'].replace(seg_map, regex=True)
#
#     # Min-Max ölçeklendirme
#     sc = MinMaxScaler((0,1))
#     df[['Days_Inactive_Last_Year','Total_Trans_Ct', 'Total_Trans_Amt']] = sc.fit_transform(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])
#
#     # KMeans model fit edildi.
#     kmeans = KMeans(n_clusters = 4, max_iter=50)
#     kmeans.fit(df[['Days_Inactive_Last_Year','Total_Trans_Ct', 'Total_Trans_Amt']])
#     df["cluster_no"] = kmeans.labels_
#     df["cluster_no"] = df["cluster_no"] + 1
#
#     # Optimum küme sayısını belirleme
#     elbow = KElbowVisualizer(KMeans(), k=(2, 20))
#     elbow.fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])
#     elbow_value = elbow.elbow_value_
#
#     # Yeni optimum küme sayısı ile model fit edilmiştir.
#     kmeans = KMeans(n_clusters = elbow_value).fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])
#     df["cluster_no"] = kmeans.labels_
#     df["cluster_no"] = df["cluster_no"] + 1
#
#     return df
#
# # RFM işlemi için önbelleğe alınmış sonucu çağırın
# df = calculate_RFM(df)
#
# st.write(df.head())

# @st.cache_data
# def create_treemap_chart(df):
#     fig = px.treemap(df, path=['Target', 'Segment'], title="Target vgizoe Segment")
#     fig.update_layout(height=600, width=800)  # Grafik boyutlarını sabitle
#     return fig
#
# # Treemap grafiğini Streamlit'e entegre et
# cached_fig = create_treemap_chart(df)
# st.plotly_chart(cached_fig)

