from catboost import CatBoostClassifier
from imblearn.under_sampling import TomekLinks
from lightgbm import LGBMClassifier
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
                             classification_report, RocCurveDisplay, roc_curve, auc, precision_recall_curve)
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
import pandas as pd
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
import warnings
warnings.simplefilter(action="ignore")
import streamlit as st
import warnings
warnings.simplefilter(action="ignore")
import pandas as pd


st.set_page_config(page_title="Model Demo", page_icon="💳", layout="wide")

st.markdown("# Churninator")
st.sidebar.header("Churninator")

#
# def grab_col_names(dataframe, cat_th=9, car_th=20):
#     #cat_cols, cat_but_car
#     cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
#     num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
#                    dataframe[col].dtypes != "O"]
#     cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
#                    dataframe[col].dtypes == "O"]
#     cat_cols = cat_cols + num_but_cat
#     cat_cols = [col for col in cat_cols if col not in cat_but_car]
#
#     #num_cols
#     num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
#     num_cols = [col for col in num_cols if col not in num_but_cat]
#
#     print(f"Observations: {dataframe.shape[0]}")
#     print(f"Variables: {dataframe.shape[1]}")
#     print(f"cat_cols: {len(cat_cols)}")
#     print(f"num_cols: {len(num_cols)}")
#     print(f"cat_but_car: {len(cat_but_car)}")
#     print(f"num_but_car: {len(num_but_cat)}")
#     return cat_cols, num_cols, cat_but_car
#
#
# def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
#     dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
#     return dataframe
#
# def combine_categories(df, cat_col1, cat_col2, new_col_name):
#     df[new_col_name] = df[cat_col1].astype(str) + '_' + df[cat_col2].astype(str)
#
# df = pd.read_csv("BankChurners.csv")
#
# df.drop([
#     "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
#     "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"],
#     inplace=True, axis=1)
#
# # Bağımlı değişkenimizin ismini target yapalım ve 1, 0 atayalım:
# df.rename(columns={"Attrition_Flag": "Target"}, inplace=True)
# df["Target"] = df.apply(lambda x: 0 if (x["Target"] == "Existing Customer") else 1, axis=1)
#
# # ID kolonunda duplicate bakıp, sonra bu değişkeni silme
# df["CLIENTNUM"].nunique()  # 10127 - yani duplicate yok id'de
# df.drop("CLIENTNUM", axis=1, inplace=True)
#
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
#
#
# # Outlier temizleme
# # IQR
# def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
#     quartile1 = dataframe[col_name].quantile(q1)
#     quartile3 = dataframe[col_name].quantile(q3)
#     interquantile_range = quartile3 - quartile1
#     up_limit = quartile3 + 1.5 * interquantile_range
#     low_limit = quartile1 - 1.5 * interquantile_range
#     return low_limit, up_limit
#
# def check_outlier(dataframe, col_name):
#     low_limit, up_limit = outlier_thresholds(dataframe, col_name)
#     if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
#         return True
#     else:
#         return False
#
# def replace_with_thresholds(dataframe, variable):
#     low_limit, up_limit = outlier_thresholds(dataframe, variable)
#     dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
#     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
#
# def grab_outliers(dataframe, col_name, index=False):
#     low, up = outlier_thresholds(dataframe, col_name)
#     print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0])
#     # if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
#     #     print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
#     # else:
#     #     print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
#
#     if index:
#         outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
#         return outlier_index
#
# for col in num_cols:
#     print(col, grab_outliers(df, col))
#
# df.shape
# #(10127, 20)
#
# def remove_outliers_from_all_columns(dataframe):
#     for col_name in num_cols:
#         low, up = outlier_thresholds(dataframe, col_name)  # Aykırı değer sınırlarını hesapla
#         outliers = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]
#         print(f"{col_name} için aykırı değer sayısı: {outliers.shape[0]}")
#         # Aykırı değerleri dataframe'den çıkar
#         dataframe = dataframe.drop(outliers.index).reset_index(drop=True)
#     return dataframe
#
# df = remove_outliers_from_all_columns(df)
# df.shape
# #(10034, 20)
#
#
#
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
# # LOF
# clf = LocalOutlierFactor(n_neighbors=20)
# clf.fit_predict(df[num_cols])
#
# df_scores = clf.negative_outlier_factor_
#
# scores = pd.DataFrame(np.sort(df_scores))
# scores.plot(stacked=True, xlim=[0, 100], style='.-')
# plt.show()
#
# th = np.sort(df_scores)[27]
#
# df.drop(axis=0, labels=df[df_scores < th].index, inplace=True) # Dropping the outliers.
# df.head()
#
# df = df.reset_index(drop=True)
# df.shape
# #(10007, 20)
#
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
# # Missing values
# cols_with_unknown = ['Income_Category', "Education_Level"]
# for col in cols_with_unknown:
#     df[col] = df[col].apply(lambda x: np.nan if x == 'Unknown' else x)
#
# # ANALİZ BAŞLANGICI
# df["On_book_cat"] = np.where((df["Months_on_book"] < 12), "<1_year", np.where((df["Months_on_book"] < 24), "<2_years", np.where((df["Months_on_book"] < 36), "<3_years", np.where((df["Months_on_book"] < 48), "<4_years", "<5_years"))))
#
#
# df['Total_Amt_Increased'] = np.where((df['Total_Amt_Chng_Q4_Q1'] > 0) & (df['Total_Amt_Chng_Q4_Q1'] < 1), 0, 1)
#
# df["Has_debt"] = np.where((df["Credit_Limit"] > df["Avg_Open_To_Buy"]), 1, 0).astype(int)
#
# df["Important_client_score"] = df["Total_Relationship_Count"] * (df["Months_on_book"] / 12)
#
# df["Avg_Trans_Amt"] = df["Total_Trans_Amt"] / df['Total_Trans_Ct']
#
# labels = ['Young', 'Middle_Aged', 'Senior']
# bins = [25, 35, 55, 74]
# df['Customer_Age_Category'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels)
#
# df["Days_Inactive_Last_Year"] = df["Months_Inactive_12_mon"] * 30
#
#
# df["Days_Inactive_Last_Year"].replace(0, 30, inplace=True)
# df["Days_Inactive_Last_Year"].replace(180, 150, inplace=True)
#
# df["RecencyScore"] = df["Days_Inactive_Last_Year"].apply(lambda x: 5 if x == 30 else
#                                                         4 if x == 60 else
#                                                         3 if x == 90 else
#                                                         2 if x == 120 else
#                                                         1 if x == 150 else x)
#
#
# df["MonetaryScore"] = pd.qcut(df["Total_Trans_Amt"], 5, labels=[1, 2, 3, 4, 5])
# df["FrequencyScore"] = pd.qcut(df["Total_Trans_Ct"], 5, labels=[1, 2, 3, 4, 5])
#
#
# combine_categories(df, 'Customer_Age_Category', 'Marital_Status', 'Age_&_Marital')
# combine_categories(df, 'Gender', 'Customer_Age_Category', 'Gender_&_Age')
# combine_categories(df, "Card_Category", "Customer_Age_Category", "Card_&_Age")
# combine_categories(df, "Gender", "FrequencyScore", "Gender_&_Frequency")
# combine_categories(df, "Gender", "MonetaryScore", "Gender_&_Monetary")
#
# df['Total_Amt_Increased'] = np.where((df['Total_Amt_Chng_Q4_Q1'] >= 0) & (df['Total_Amt_Chng_Q4_Q1'] < 1), 0, 1)
# df['Total_Ct_Increased'] = np.where((df['Total_Ct_Chng_Q4_Q1'] >= 0) & (df['Total_Ct_Chng_Q4_Q1'] < 1), 0, 1)
#
#
# df.loc[(df["Total_Ct_Chng_Q4_Q1"] == 1) & (df["Total_Amt_Chng_Q4_Q1"] > 1), "Ct_vs_Amt"] = "Same_ct_inc_amt"
# # boş
# df.loc[(df["Total_Ct_Chng_Q4_Q1"] == 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Same_ct_same_amt" # BOŞ
# # İşlem sayısı aynı kalıp, harcama miktarı azalanlar: (harcamalardan mı kısıyorlar? belki ihtiyaçları olanları almışlardır.) TODO May_Marry ile incele)
# df.loc[(df["Total_Ct_Chng_Q4_Q1"] == 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Same_ct_dec_amt"
# # işlem sayısı da, miktarı da artmış (bizi sevindiren müşteri <3 )
# df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] > 1), "Ct_vs_Amt"] = "Inc_ct_inc_amt"
# # BOŞ İşlem sayısı artmasına rağmen, harcama miktarı aynı kalanlar: (aylık ortalama harcama azalıyor)
# df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Inc_ct_same_amt" # BOŞ
# # İşlem sayısı artmış ama miktar azalmış. Yani daha sık, ama daha küçük alışverişler yapıyor. Bunlar düşük income grubuna aitse bankayı mutlu edecek bir davranış.
# df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Inc_ct_dec_amt"
# #(df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1)]).groupby("Income_Category").count() # Evet, düşük income grubuna ait.
# # İşlem sayısı azalmış ama daha büyük miktarlarda harcama yapılıyor:
# df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] > 1), "Ct_vs_Amt"] = "Dec_ct_inc_amt"
# # İşlem sayısı azalmış, toplam miktar aynı kalmış (yani ortalama harcama artmış):
# df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Dec_ct_same_amt"
# df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Dec_ct_dec_amt"
#
#
# # Personalar
# df["Affluent_criteria"] = (df['Income_Category'] == '$120K +').astype("Int64")
# df["Budget_criteria"] = ((df['Income_Category'] == 'Less than $40K') & (df['Education_Level'].isin(['High School', 'College']))).astype("Int64")
# df["Young_prof_criteria"] = ((df['Customer_Age'] <= 30) & (df['Education_Level'].isin(['College', 'Graduate']))).astype("Int64")
# df["Family_criteria"] = ((df["Marital_Status"].isin(["Married", "Divorced", "Unknown"])) & (df['Dependent_count'] >= 3)).astype(int)
# df["Credit_builder_criteria"] = (df['Credit_Limit'] < 2500).astype(int)  # This threshold is chosen to include individuals with credit limits in the lower percentiles of the distribution, which may indicate a need for credit-building strategies or entry-level credit products.
# df["Digital_criteria"] = (df['Contacts_Count_12_mon'] == 0).astype(int)
# df["High_net_worth_individual"] = ((df['Income_Category'] == '$120K +') & (df['Total_Trans_Amt'] > 5000)).astype("Int64")
# df["Rewards_maximizer"] = ((df['Total_Trans_Amt'] > 10000) & (df['Total_Revolving_Bal'] == 0)).astype(int) # For the Rewards_maximizer column, the threshold for Total_Trans_Amt is also set at $10000. Since rewards maximizers are individuals who strategically maximize rewards and benefits from credit card usage, it's reasonable to expect that they engage in higher levels of spending. Therefore, the threshold of $10000 for Total_Trans_Amt appears appropriate for identifying rewards maximizers, considering that it captures individuals with relatively high spending habits.
# df["May_marry"] = ((df["Age_&_Marital"] == "Young_Single") & (df['Dependent_count'] == 0)).astype(int)
#
# df["Product_by_Year"] = df["Total_Relationship_Count"] / (df["Months_on_book"] / 12)
#
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
#
# # Rare analyser:
# def rare_analyser(dataframe, target, cat_cols):
#     for col in cat_cols:
#         print(col, ":", len(dataframe[col].value_counts()))
#         print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
#                             "RATIO": dataframe[col].value_counts() / len(dataframe),
#                             "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
#
# rare_analyser(df, "Target", cat_cols)
#
# # Rare encoding:
# df["Card_Category"] = df["Card_Category"].apply(lambda x: "Gold_Platinum" if x == "Platinum" or x == "Gold" else x)
# df["Months_Inactive_12_mon"] = df["Months_Inactive_12_mon"].apply(lambda x: 1 if x == 0 else x)
# df["Ct_vs_Amt"] = df["Ct_vs_Amt"].apply(lambda x: "Dec_ct_inc_amt" if x == "Dec_ct_same_amt" else x)
# df["Ct_vs_Amt"] = df["Ct_vs_Amt"].apply(lambda x: "Inc_ct_inc_amt" if x == "Same_ct_inc_amt" else x)
# df["Contacts_Count_12_mon"] = df["Contacts_Count_12_mon"].apply(lambda x: 5 if x == 6 else x)
# df["Card_&_Age"] = df["Card_&_Age"].apply(lambda x: "Rare" if df["Card_&_Age"].value_counts()[x] < 30 else x)
# df["Card_&_Age"].value_counts()
#
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
#
# categories_dict = {
#         "Education_Level": ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', np.nan],
#         "Income_Category": ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', np.nan],
#         "Customer_Age_Category": ['Young', 'Middle_Aged', 'Senior'],
#         "Card_Category": ['Blue', 'Silver', 'Gold_Platinum'],
#         "On_book_cat": ["<2_years", "<3_years", "<4_years", "<5_years"]}
#
# def ordinal_encoder(dataframe, col):
#     if col in categories_dict:
#         col_cats = categories_dict[col]
#         ordinal_encoder = OrdinalEncoder(categories=[col_cats])
#         dataframe[col] = ordinal_encoder.fit_transform(dataframe[[col]])
#
#     return dataframe
#
# for col in df.columns:
#     ordinal_encoder(df, col)
#
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
# df = one_hot_encoder(df, ["Gender"], drop_first=True) # M'ler 1.
# df.rename(columns={"Gender_M": "Gender"}, inplace=True)
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
# df.isnull().sum()
# df.shape # (10102, 47)
#
# #knn eski
# numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
# categorical_columns = [col for col in df.columns if col not in numeric_columns]
# df_numeric = df[numeric_columns]
# imputer = KNNImputer(n_neighbors=10)
# df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_columns)
# df = pd.concat([df_numeric_imputed, df[categorical_columns]], axis=1)
# df["Education_Level"] = df["Education_Level"].round().astype("Int64")
# df["Income_Category"] = df["Income_Category"].round().astype("Int64")
#
#
# df.isnull().sum()
# df.shape
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
#
# # One-hot encoding:
# df = one_hot_encoder(df, ["Marital_Status",
#                           "Age_&_Marital",
#                           "Gender_&_Age",
#                           "Card_&_Age",
#                           "Gender_&_Frequency",
#                           "Gender_&_Monetary",
#                           'Ct_vs_Amt',
#                           'Dependent_count',
#                           'Total_Relationship_Count',
#                           'Months_Inactive_12_mon',
#                           'Contacts_Count_12_mon'],
#                           drop_first=True)
#
# # Gizemin yarattığı ve belki onehot'a girecek kolonlar:
# # 'Year_on_book', "RFM_SCORE", Segment, Cluster, RFMSegment, cluster_no
#
# useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
#                 (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
# df.drop(useless_cols, axis=1, inplace=True)
#
# df.head()
# dff = df.copy()
# df = dff.copy()
#
#
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
# # Değişken tipi dönüştürme:
# for col in df.columns:
#     if df[col].dtype == 'float64':  # Sadece float sütunları kontrol edelim
#         if (df[col] % 1 == 000).all():  # Tüm değerlerin virgülden sonrası 0 mı kontrol edelim
#             df[col] = df[col].astype(int)
#
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
#
#
# """rfm skorları ile segmentasyon oluşturma"""
# # Total_Trans_Amt = Monetary
# # Total_Trans_Ct = Frequency
# # Days_Inactive_Last_Year  Recency
#
# # Recency: A recent purchase indicates that the customer is active and potentially more receptive to further
# # communication or offers.
#
# seg_map = {
#         r'[1-2][1-2]': 'Hibernating',
#         r'[1-2][3-4]': 'At Risk',
#         r'[1-2]5': 'Can\'t Lose',
#         r'3[1-2]': 'About to Sleep',
#         r'33': 'Need Attention',
#         r'[3-4][4-5]': 'Loyal Customers',
#         r'41': 'Promising',
#         r'51': 'New Customers',
#         r'[4-5][2-3]': 'Potential Loyalists',
#         r'5[4-5]': 'Champions'
# }
#
# # segment oluşturma (Recency + Frequency)
# df['Segment'] = df['RecencyScore'].astype(str) + df['FrequencyScore'].astype(str)
# df['Segment'] = df['Segment'].replace(seg_map, regex=True)
# df.head(40)
#
#
# # Feature scaling (robust):
# # TODO GBM için scale etmeden deneyeceğiz.
# rs = RobustScaler()
# df[num_cols] = rs.fit_transform(df[num_cols])
# df[["Days_Inactive_Last_Year"]] = rs.fit_transform(df[["Days_Inactive_Last_Year"]])
#
#
# from sklearn.cluster import KMeans
# # model fit edildi.
# kmeans = KMeans(n_clusters=4, max_iter=50, random_state=1)
# kmeans.fit(df[['Days_Inactive_Last_Year','Total_Trans_Ct', 'Total_Trans_Amt']])
#
# df["cluster_no"] = kmeans.labels_
# df["cluster_no"] = df["cluster_no"] + 1
# df.groupby("cluster_no")["Segment"].value_counts()
#
# ssd = []
#
# K = range(1, 30)
#
# for k in K:
#     kmeans = KMeans(n_clusters=k, random_state=1).fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])
#     ssd.append(kmeans.inertia_) #inertia her bir k değeri için ssd değerini bulur.
#
#
#
# # Optimum küme sayısını belirleme
# from yellowbrick.cluster import KElbowVisualizer
# kmeans = KMeans(random_state=1)
# elbow = KElbowVisualizer(kmeans, k=(2, 20))
# elbow.fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])
# elbow.show()
# elbow.elbow_value_
#
# # yeni optimum kümse sayısı ile model fit edilmiştir.
# kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=1).fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])
#
#
# # Cluster_no 0'dan başlamaktadır. Bunun için 1 eklenmiştir.
# df["cluster_no"] = kmeans.labels_
# df["cluster_no"] = df["cluster_no"] + 1
#
# df.groupby("cluster_no")["Segment"].value_counts()
# # cluster_no  Segment
# # 1           Potential Loyalists     748 4
# #             Promising               693
# #             New Customers           424
# # 2           Potential Loyalists    1321
# #             Loyal Customers         849  7
# #             Champions               609
# #             Promising                 2
# # 3           Loyal Customers         366  5
# #             Champions               136
# #             Need Attention           64
# #             Potential Loyalists      59
# #             About to Sleep           16
# #             At Risk                   4
# #             Hibernating               1
# # 4           Loyal Customers         510  6
# #             Champions               204
# #             Can't Lose               31
# # 5           At Risk                 302  1
# #             Hibernating             160
# #             Can't Lose               85
# # 6           About to Sleep         1471  2
# #             Hibernating             146
# #             Need Attention           37
# # 7           Loyal Customers         915 3
# #             Need Attention          680
# #             About to Sleep          174
#
# df['cluster_no'] = df['cluster_no'].replace({1: 4, 2: 7, 3: 5, 4: 6, 5: 1, 6: 2, 7: 3})
#
# print(df)
#
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
# df[num_cols].head()
# #33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
#
#
#
# # Korelasyon Heatmap:
# def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
#     corr = dataframe[num_cols].corr()
#     cor_matrix = corr.abs()
#     upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
#     drop_list = [col for col in upper_triangle_matrix if any(upper_triangle_matrix[col] > corr_th)]
#     if plot:
#         sns.set(rc={'figure.figsize': (12, 12)})
#         sns.heatmap(corr, cmap='RdBu', annot= True, vmin=-1, vmax=1)
#         plt.show()
#     return drop_list
#
#
# drop_list = high_correlated_cols(df, plot=True)
#
# df.drop(columns=drop_list, inplace=True, axis=1)
#
# df.shape
# #(10007, 96)
#
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
#
#
# df[['MonetaryScore', 'FrequencyScore']] = df[['MonetaryScore', 'FrequencyScore']].astype(int)
#
# dff = df.copy()
# df.head()
# y = df["Target"]
# X = df.drop(["Target", "Segment"], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Undersampling (Tomek Links)
# # summarize class distribution
# counter = Counter(y)
# print(counter) # {0: 8397, 1: 1610}
# # define the undersampling method
# undersample = TomekLinks()
# # transform the dataset
# X_train, y_train = undersample.fit_resample(X_train, y_train)
# # summarize the new class distribution
# counter = Counter(y_train)
# print(counter) # {0: 8254, 1: 1610}
#
# def model_metrics(X_train, y_train, X_test, y_test):
#     print("Base Models....")
#     classifiers = [('LR', LogisticRegression()),
#                    ('KNN', KNeighborsClassifier()),
#                    ("SVC", SVC()),
#                    ("CART", DecisionTreeClassifier()),
#                    ("RF", RandomForestClassifier()),
#                    ('Adaboost', AdaBoostClassifier()),
#                    ('GBM', GradientBoostingClassifier()),
#                    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
#                    ('LightGBM', LGBMClassifier()),
#                    ('CatBoost', CatBoostClassifier(verbose=False))
#                    ]
#
#     for name, classifier in classifiers:
#         model = classifier.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#
#         # Classification Report
#         report = classification_report(y_test, y_pred)
#         print(f"Classification Report for {name}:")
#         print(report)
#
# model_metrics(X_train, y_train, X_test, y_test)
#
# # undersampling'siz
# # Base Models....
# # Classification Report for LR:
# #               precision    recall  f1-score   support
# #            0       0.96      0.97      0.96      1694
# #            1       0.82      0.78      0.80       308
# #     accuracy                           0.94      2002
# #    macro avg       0.89      0.88      0.88      2002
# # weighted avg       0.94      0.94      0.94      2002
# # Classification Report for KNN:
# #               precision    recall  f1-score   support
# #            0       0.95      0.98      0.96      1694
# #            1       0.85      0.69      0.76       308
# #     accuracy                           0.93      2002
# #    macro avg       0.90      0.84      0.86      2002
# # weighted avg       0.93      0.93      0.93      2002
# # Classification Report for SVC:
# #               precision    recall  f1-score   support
# #            0       0.95      0.98      0.97      1694
# #            1       0.86      0.73      0.79       308
# #     accuracy                           0.94      2002
# #    macro avg       0.91      0.85      0.88      2002
# # weighted avg       0.94      0.94      0.94      2002
# # Classification Report for CART:
# #               precision    recall  f1-score   support
# #            0       0.97      0.95      0.96      1694
# #            1       0.75      0.84      0.80       308
# #     accuracy                           0.93      2002
# #    macro avg       0.86      0.90      0.88      2002
# # weighted avg       0.94      0.93      0.93      2002
# # Classification Report for RF:
# #               precision    recall  f1-score   support
# #            0       0.97      0.99      0.98      1694
# #            1       0.93      0.81      0.86       308
# #     accuracy                           0.96      2002
# #    macro avg       0.95      0.90      0.92      2002
# # weighted avg       0.96      0.96      0.96      2002
# # Classification Report for Adaboost:
# #               precision    recall  f1-score   support
# #            0       0.97      0.98      0.97      1694
# #            1       0.86      0.82      0.84       308
# #     accuracy                           0.95      2002
# #    macro avg       0.91      0.90      0.91      2002
# # weighted avg       0.95      0.95      0.95      2002
# # Classification Report for GBM:
# #               precision    recall  f1-score   support
# #            0       0.97      0.99      0.98      1694
# #            1       0.93      0.83      0.88       308
# #     accuracy                           0.97      2002
# #    macro avg       0.95      0.91      0.93      2002
# # weighted avg       0.96      0.97      0.96      2002
# # Classification Report for XGBoost:
# #               precision    recall  f1-score   support
# #            0       0.98      0.98      0.98      1694
# #            1       0.91      0.89      0.90       308
# #     accuracy                           0.97      2002
# #    macro avg       0.95      0.94      0.94      2002
# # weighted avg       0.97      0.97      0.97      2002
# # [LightGBM] [Info] Number of positive: 1302, number of negative: 6703
# # [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.002116 seconds.
# # You can set `force_row_wise=true` to remove the overhead.
# # And if memory is not enough, you can set `force_col_wise=true`.
# # [LightGBM] [Info] Total Bins 2295
# # [LightGBM] [Info] Number of data points in the train set: 8005, number of used features: 94
# # [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.162648 -> initscore=-1.638654
# # [LightGBM] [Info] Start training from score -1.638654
# # Classification Report for LightGBM:
# #               precision    recall  f1-score   support
# #            0       0.98      0.99      0.98      1694
# #            1       0.92      0.91      0.91       308
# #     accuracy                           0.97      2002
# #    macro avg       0.95      0.95      0.95      2002
# # weighted avg       0.97      0.97      0.97      2002
# # Classification Report for CatBoost:
# #               precision    recall  f1-score   support
# #            0       0.98      0.99      0.98      1694
# #            1       0.92      0.90      0.91       308
# #     accuracy                           0.97      2002
# #    macro avg       0.95      0.94      0.95      2002
# # weighted avg       0.97      0.97      0.97      2002
#
# # undersampling'li
# # Base Models....
# # Classification Report for LR:
# #               precision    recall  f1-score   support
# #            0       0.96      0.97      0.96      1694
# #            1       0.81      0.80      0.81       308
# #     accuracy                           0.94      2002
# #    macro avg       0.89      0.88      0.89      2002
# # weighted avg       0.94      0.94      0.94      2002
# # Classification Report for KNN:
# #               precision    recall  f1-score   support
# #            0       0.95      0.97      0.96      1694
# #            1       0.82      0.71      0.76       308
# #     accuracy                           0.93      2002
# #    macro avg       0.89      0.84      0.86      2002
# # weighted avg       0.93      0.93      0.93      2002
# # Classification Report for SVC:
# #               precision    recall  f1-score   support
# #            0       0.96      0.98      0.97      1694
# #            1       0.85      0.76      0.80       308
# #     accuracy                           0.94      2002
# #    macro avg       0.90      0.87      0.88      2002
# # weighted avg       0.94      0.94      0.94      2002
# # Classification Report for CART:
# #               precision    recall  f1-score   support
# #            0       0.97      0.95      0.96      1694
# #            1       0.75      0.83      0.79       308
# #     accuracy                           0.93      2002
# #    macro avg       0.86      0.89      0.87      2002
# # weighted avg       0.94      0.93      0.93      2002
# # Classification Report for RF:
# #               precision    recall  f1-score   support
# #            0       0.97      0.98      0.98      1694
# #            1       0.90      0.84      0.87       308
# #     accuracy                           0.96      2002
# #    macro avg       0.93      0.91      0.92      2002
# # weighted avg       0.96      0.96      0.96      2002
# # Classification Report for Adaboost:
# #               precision    recall  f1-score   support
# #            0       0.97      0.97      0.97      1694
# #            1       0.85      0.84      0.85       308
# #     accuracy                           0.95      2002
# #    macro avg       0.91      0.91      0.91      2002
# # weighted avg       0.95      0.95      0.95      2002
# # Classification Report for GBM:
# #               precision    recall  f1-score   support
# #            0       0.97      0.99      0.98      1694
# #            1       0.93      0.85      0.89       308
# #     accuracy                           0.97      2002
# #    macro avg       0.95      0.92      0.93      2002
# # weighted avg       0.97      0.97      0.97      2002
# # Classification Report for XGBoost:
# #               precision    recall  f1-score   support
# #            0       0.98      0.98      0.98      1694
# #            1       0.89      0.91      0.90       308
# #     accuracy                           0.97      2002
# #    macro avg       0.94      0.95      0.94      2002
# # weighted avg       0.97      0.97      0.97      2002
# # [LightGBM] [Info] Number of positive: 1302, number of negative: 6592
# # [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.002157 seconds.
# # You can set `force_row_wise=true` to remove the overhead.
# # And if memory is not enough, you can set `force_col_wise=true`.
# # [LightGBM] [Info] Total Bins 2295
# # [LightGBM] [Info] Number of data points in the train set: 7894, number of used features: 94
# # [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.164935 -> initscore=-1.621955
# # [LightGBM] [Info] Start training from score -1.621955
# # Classification Report for LightGBM:
# #               precision    recall  f1-score   support
# #            0       0.98      0.98      0.98      1694
# #            1       0.91      0.89      0.90       308
# #     accuracy                           0.97      2002
# #    macro avg       0.94      0.94      0.94      2002
# # weighted avg       0.97      0.97      0.97      2002
# # Classification Report for CatBoost:
# #               precision    recall  f1-score   support
# #            0       0.98      0.99      0.98      1694
# #            1       0.92      0.90      0.91       308
# #     accuracy                           0.97      2002
# #    macro avg       0.95      0.94      0.95      2002
# # weighted avg       0.97      0.97      0.97      2002
#
# ################################################################################
# svc_params = {
#     'C': [1, 10],
#     'gamma': [0.01, 0.1, "scale"],
#     'kernel': ['rbf']
# }
# adaboost_params = {
#     'n_estimators': [50, 100],
#     'learning_rate': [0.1, 0.5],
#     'algorithm': ['SAMME.R']
# }
# knn_params = {
#     "n_neighbors": range(5, 20, 5),
#     'weights': ['uniform', 'distance'],
#     'algorithm': ['auto'],
#     'p': [1, 2]
# }
# cart_params = {
#     'max_depth': [5, 10],
#     "min_samples_split": range(10, 20, 5)
# }
# gbm_params = {
#     "learning_rate": [0.1],
#     "max_depth": [3, 8],
#     "n_estimators": [100, 200],
#     "subsample": [0.7]
# }
# rf_params = {
#     "max_depth": [None, 10],
#     "max_features": ["sqrt"],
#     "min_samples_split": [2, 10],
#     "n_estimators": [100, 200]
# }
# xgboost_params = {
#     "learning_rate": [0.1],
#     "max_depth": [3, 7],
#     "n_estimators": [100, 200],
#     "colsample_bytree": [0.7]
# }
#
# xgboost_params = {
#     "learning_rate": [0.05, 0.1, 0.2],  # Varsayılan değer 0.1'dir, bu yüzden etrafına genişletme yapıldı.
#     "max_depth": [3, 5, 7],  # Varsayılan değer 6'dır, ancak daha düşük ve biraz yüksek değerlerle genişletildi.
#     "n_estimators": [100, 200, 300],  # Varsayılan değer 100'dür, yüksek iterasyon sayıları eklendi.
#     "colsample_bytree": [0.7, 1.0],  # Varsayılan değer 1'dir. Çeşitlilik sağlamak için 0.7 ekledik.
#     "subsample": [0.7, 1.0]  # Örneklem alt kümesini denetlemek için ek parametre.
# }
#
# lightgbm_params = {
#     "learning_rate": [0.1],
#     "n_estimators": [100, 200],
#     "max_depth": [3, 7],
#     "colsample_bytree": [0.7]
# }
# catboost_params = {
#     "learning_rate": [0.1],
#     "depth": [6],
#     "iterations": [100, 200],
#     "subsample": [0.75]
# }
# logistic_params = {
#     'penalty': ['l2'],
#     'C': [1, 10],
#     'solver': ['liblinear', 'lbfgs'],
#     'max_iter': [100, 200],
#     'class_weight': [None, 'balanced']
# }
#
#
#
# ##############################################################################
# classifiers = [#('Adaboost', AdaBoostClassifier(), adaboost_params),
#     #('KNN', KNeighborsClassifier(), knn_params),
#     #("CART", DecisionTreeClassifier(), cart_params),
#     #("RF", RandomForestClassifier(), rf_params),
#     #("LogisticRegression", LogisticRegression(), logistic_params),  # Lojistik Regresyon
#     #("SVC", SVC(), svc_params),  # Destek Vektör Makineleri
#     #("GBM", GradientBoostingClassifier(), gbm_params),  # Gradyan Arttırma Makineleri
#     ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
#     #('LightGBM', LGBMClassifier(force_col_wise=True), lightgbm_params),
#     #('CatBoost', CatBoostClassifier(verbose=False), catboost_params)
# ]
#
#
# def hyperparameter_optimization(X_train, y_train, X_test, y_test, cv=5, scoring="recall"):
#     print("Hyperparameter Optimization....")
#     best_models = {}
#     for name, classifier, params in classifiers:
#         print(f"########## {name} ##########")
#         cv_results = cross_validate(classifier, X_train, y_train, cv=cv, scoring=scoring)
#         print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")
#
#         gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X_train, y_train)
#         final_model = classifier.set_params(**gs_best.best_params_)
#
#         cv_results = cross_validate(final_model, X_train, y_train, cv=cv, scoring=scoring)
#         print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
#         print(f"{name} best params: {gs_best.best_params_}")
#
#         # Test verileri üzerinde modelin performansını değerlendir
#         final_model.fit(X_train, y_train)
#         y_pred = final_model.predict(X_test)
#         report = classification_report(y_test, y_pred)
#         print(f"{name} classification report:\n{report}\n")
#
#         best_models[name] = final_model
#
#     return best_models, gs_best.best_params_, y_pred
#
# model, best_params, y_pred = hyperparameter_optimization(X_train, y_train, X_test, y_test)
# # Hyperparameter Optimization....
# # ########## Adaboost ##########
# # recall (Before): 0.8418
# # recall (After): 0.831
# # Adaboost best params: {'algorithm': 'SAMME.R', 'learning_rate': 0.5, 'n_estimators': 100}
# # Adaboost classification report:
# #               precision    recall  f1-score   support
# #            0       0.97      0.98      0.97      1694
# #            1       0.87      0.82      0.85       308
# #     accuracy                           0.95      2002
# #    macro avg       0.92      0.90      0.91      2002
# # weighted avg       0.95      0.95      0.95      2002
# # ########## KNN ##########
# # recall (Before): 0.6605
# # recall (After): 0.682
# # KNN best params: {'algorithm': 'auto', 'n_neighbors': 10, 'p': 1, 'weights': 'distance'}
# # KNN classification report:
# #               precision    recall  f1-score   support
# #            0       0.94      0.97      0.96      1694
# #            1       0.81      0.67      0.74       308
# #     accuracy                           0.93      2002
# #    macro avg       0.88      0.82      0.85      2002
# # weighted avg       0.92      0.93      0.92      2002
# # ########## CART ##########
# # recall (Before): 0.811
# # recall (After): 0.7949
# # CART best params: {'max_depth': 10, 'min_samples_split': 10}
# # CART classification report:
# #               precision    recall  f1-score   support
# #            0       0.97      0.96      0.96      1694
# #            1       0.80      0.82      0.81       308
# #     accuracy                           0.94      2002
# #    macro avg       0.88      0.89      0.89      2002
# # weighted avg       0.94      0.94      0.94      2002
# # ########## RF ##########
# # recall (Before): 0.7888
# # recall (After): 0.798
# # RF best params: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 200}
# # RF classification report:
# #               precision    recall  f1-score   support
# #            0       0.97      0.99      0.98      1694
# #            1       0.93      0.83      0.88       308
# #     accuracy                           0.97      2002
# #    macro avg       0.95      0.91      0.93      2002
# # weighted avg       0.96      0.97      0.96      2002
#
# ########## LogisticRegression ##########
# # recall (Before): 0.7765
# # recall (After): 0.7757
# # LogisticRegression best params: {'C': 1, 'class_weight': None, 'max_iter': 200, 'penalty': 'l2', 'solver': 'lbfgs'}
# # LogisticRegression classification report:
# #               precision    recall  f1-score   support
# #            0       0.96      0.97      0.96      1694
# #            1       0.82      0.78      0.80       308
# #     accuracy                           0.94      2002
# #    macro avg       0.89      0.88      0.88      2002
# # weighted avg       0.94      0.94      0.94      2002
# # ########## SVC ##########
# # recall (Before): 0.745
# # recall (After): 0.8134
# # SVC best params: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
# # SVC classification report:
# #               precision    recall  f1-score   support
# #            0       0.97      0.98      0.97      1694
# #            1       0.86      0.82      0.84       308
# #     accuracy                           0.95      2002
# #    macro avg       0.92      0.90      0.91      2002
# # weighted avg       0.95      0.95      0.95      2002
# # ########## GBM ##########
# # recall (Before): 0.8433
# # recall (After): 0.8725
# # GBM best params: {'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 200, 'subsample': 0.7}
# # GBM classification report:
# #               precision    recall  f1-score   support
# #            0       0.98      0.99      0.99      1694
# #            1       0.94      0.90      0.92       308
# #     accuracy                           0.97      2002
# #    macro avg       0.96      0.94      0.95      2002
# # weighted avg       0.97      0.97      0.97      2002
# # ########## XGBoost ##########
# # recall (Before): 0.8817
# # recall (After): 0.8825
# # XGBoost best params: {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200}
# # XGBoost classification report:
# #               precision    recall  f1-score   support
# #            0       0.98      0.99      0.98      1694
# #            1       0.92      0.91      0.91       308
# #     accuracy                           0.97      2002
# #    macro avg       0.95      0.95      0.95      2002
# # weighted avg       0.97      0.97      0.97      2002
#
# # ########## LightGBM ##########
# # LightGBM classification report:
# #               precision    recall  f1-score   support
# #            0       0.98      0.99      0.99      1694
# #            1       0.94      0.89      0.92       308
# #     accuracy                           0.97      2002
# #    macro avg       0.96      0.94      0.95      2002
# # weighted avg       0.97      0.97      0.97      2002
# # ########## CatBoost ##########
# # recall (Before): 0.8817
# # recall (After): 0.8833
# # CatBoost best params: {'depth': 6, 'iterations': 200, 'learning_rate': 0.1, 'subsample': 0.75}
# # CatBoost classification report:
# #               precision    recall  f1-score   support
# #            0       0.98      0.99      0.98      1694
# #            1       0.92      0.88      0.90       308
# #     accuracy                           0.97      2002
# #    macro avg       0.95      0.93      0.94      2002
# # weighted avg       0.97      0.97      0.97      2002
#
#
# # XGBoost Final modelini oluşturun
# final_model = XGBClassifier(**best_params)
#
# # Modeli eğitin
# final_model.fit(X_train, y_train)
#
# # Train verisi için tahmin olasılıklarını alın
# train_proba = final_model.predict_proba(X_train)[:, 1]
#
# # Test verisi için tahmin olasılıklarını alın
# test_proba = final_model.predict_proba(X_test)[:, 1]
#
# # Train verisi için ROC eğrisini hesaplayın
# train_fpr, train_tpr, _ = roc_curve(y_train, train_proba)
# train_auc = auc(train_fpr, train_tpr)
#
# # Test verisi için ROC eğrisini hesaplayın
# test_fpr, test_tpr, _ = roc_curve(y_test, test_proba)
# test_auc = auc(test_fpr, test_tpr)
#
# # Eğitim ve test ROC eğrilerini çiz
# plt.figure(figsize=(8, 6))
# plt.plot(train_fpr, train_tpr, color='blue', lw=2, label=f'Eğitim ROC (AUC = {train_auc:.2f})')
# plt.plot(test_fpr, test_tpr, color='green', lw=2, label=f'Test ROC (AUC = {test_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Yanlış Pozitif Oranı (FPR)')
# plt.ylabel('Doğru Pozitif Oranı (TPR)')
# plt.title('Eğitim ve Test ROC Eğrileri')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()
#
# #presicion-recall eğrisi:
# precision, recall, _ = precision_recall_curve(y_test, test_proba)
#
# # Precision-Recall eğrisini çiz
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, color='blue', lw=2)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Eğrisi')
# plt.grid(True)
# plt.show()
#
# # Detaylı sınıflandırma raporunu alın
# y_pred = final_model.predict(X_test)
# report = classification_report(y_test, y_pred, target_names=['Negatif', 'Pozitif'])
# print(report)
#
#
#
#
#
#
# ################################
# # Feature Importance
# ################################
# def plot_importance(model, features, num=len(X), save=False):
#     feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
#     plt.figure(figsize=(10, 10))
#     sns.set(font_scale=1)
#     sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
#                                                                      ascending=False)[0:num])
#     plt.title('Features')
#     plt.tight_layout()
#     plt.show()
#     if save:
#         plt.savefig('importances.png')
#
# plot_importance(final_model, X)
#
#
# #top 15
# plot_importance(final_model, X, num=15)
#
#
# df.head()
# df.shape
# X.head()
# X.shape
#
#
#
# #y_pred y_test segment birleştirme
# # 'Segment' bilgisini kaydet
# segment_info = df['Segment']
#
#
# # Test seti indekslerini al
# test_indices = X_test.index
#
# # Test seti indekslerine göre 'Segment' bilgisini çek
# test_segments = segment_info[test_indices]
#
# # Sonuç DataFrame'i oluştur
# results_df = pd.DataFrame({
#     'Tahmin': y_pred,
#     'Gerçek Değer': y_test,
#     'Segment': test_segments
# })
#
#
# results_df.head()
# results_df.shape
# # Hem gerçek değerlerin hem de tahminlerin 1 olduğu durumları filtreleme
# correct_and_one = results_df[(results_df['Gerçek Değer'] == 1) & (results_df['Tahmin'] == 1)]
#
# # Bu durumlara ait Segment değerlerinin sayısını hesaplama
# segment_counts_one = correct_and_one['Segment'].value_counts()

data = {
    'Segment': ['About to Sleep', 'Potential Loyalists', 'Promising', 'Hibernating', 'Need Attention',
                'New Customers', 'Loyal Customers', 'At Risk', 'Champions', "Can't Lose"],
    'Count': [124, 45, 38, 32, 14, 11, 11, 2, 1, 1]
}

# DataFrame oluşturma
segment_df = pd.DataFrame(data)
percentages = [124, 45, 38, 32, 14, 11, 11, 2, 1, 1]
total = 279
formatted_percentages = [f"{(count / total * 100):.1f}%" for count in percentages]

# Creating the formatted string for streamlit
formatted_string = f"""
**Bu gruplar**:
- **About to Sleep**: {formatted_percentages[0]}
- **Potential Loyalists**: {formatted_percentages[1]}
- **Promising**: {formatted_percentages[2]}
- **Hibernating**: {formatted_percentages[3]}
- **Need Attention**: {formatted_percentages[4]}
- **New Customers**: {formatted_percentages[5]}
- **Loyal Customers**: {formatted_percentages[6]}
- **At Risk**: {formatted_percentages[7]}
- **Champions**: {formatted_percentages[8]}
- **Can't Lose**: {formatted_percentages[9]}
"""

st.markdown(
    """
    <style>
    .button {
        background-color: purple !important;
    }
    .button:hover {
        background-color: blue !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
result = st.button("Müşterileri Bilgilerini Serverdan Çekmek İçin Tıklayın")
if result:
  st.write("Kaybetme riski ile karşı karşıya olduğunuz müşterilerinizi on farklı gruba ayırdık.")
  st.write(formatted_string)
  st.write("**Önerilerimiz:**")
  st.write()
