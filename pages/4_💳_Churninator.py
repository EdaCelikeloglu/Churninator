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

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

st.set_page_config(page_title="Model Demo", page_icon="💳", layout="wide")

st.markdown("# Churninator")
st.sidebar.header("Churninator")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

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

df = pd.read_csv("BankChurners.csv")

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

df.shape
#(10127, 20)

def remove_outliers_from_all_columns(dataframe):
    for col_name in num_cols:
        low, up = outlier_thresholds(dataframe, col_name)  # Aykırı değer sınırlarını hesapla
        outliers = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]
        print(f"{col_name} için aykırı değer sayısı: {outliers.shape[0]}")
        # Aykırı değerleri dataframe'den çıkar
        dataframe = dataframe.drop(outliers.index).reset_index(drop=True)
    return dataframe

df = remove_outliers_from_all_columns(df)
df.shape
#(10034, 20)



cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LOF
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols])

df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 100], style='.-')
plt.show()

th = np.sort(df_scores)[27]

df.drop(axis=0, labels=df[df_scores < th].index, inplace=True) # Dropping the outliers.
df.head()

df = df.reset_index(drop=True)
df.shape
#(10007, 20)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Missing values
cols_with_unknown = ['Income_Category', "Education_Level"]
for col in cols_with_unknown:
    df[col] = df[col].apply(lambda x: np.nan if x == 'Unknown' else x)

# ANALİZ BAŞLANGICI
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
df.loc[(df["Total_Ct_Chng_Q4_Q1"] == 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Same_ct_same_amt" # BOŞ
# İşlem sayısı aynı kalıp, harcama miktarı azalanlar: (harcamalardan mı kısıyorlar? belki ihtiyaçları olanları almışlardır.) TODO May_Marry ile incele)
df.loc[(df["Total_Ct_Chng_Q4_Q1"] == 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Same_ct_dec_amt"
# işlem sayısı da, miktarı da artmış (bizi sevindiren müşteri <3 )
df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] > 1), "Ct_vs_Amt"] = "Inc_ct_inc_amt"
# BOŞ İşlem sayısı artmasına rağmen, harcama miktarı aynı kalanlar: (aylık ortalama harcama azalıyor)
df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Inc_ct_same_amt" # BOŞ
# İşlem sayısı artmış ama miktar azalmış. Yani daha sık, ama daha küçük alışverişler yapıyor. Bunlar düşük income grubuna aitse bankayı mutlu edecek bir davranış.
df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Inc_ct_dec_amt"
#(df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1)]).groupby("Income_Category").count() # Evet, düşük income grubuna ait.
# İşlem sayısı azalmış ama daha büyük miktarlarda harcama yapılıyor:
df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] > 1), "Ct_vs_Amt"] = "Dec_ct_inc_amt"
# İşlem sayısı azalmış, toplam miktar aynı kalmış (yani ortalama harcama artmış):
df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Dec_ct_same_amt"
df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Dec_ct_dec_amt"


# Personalar
df["Affluent_criteria"] = (df['Income_Category'] == '$120K +').astype("Int64")
df["Budget_criteria"] = ((df['Income_Category'] == 'Less than $40K') & (df['Education_Level'].isin(['High School', 'College']))).astype("Int64")
df["Young_prof_criteria"] = ((df['Customer_Age'] <= 30) & (df['Education_Level'].isin(['College', 'Graduate']))).astype("Int64")
df["Family_criteria"] = ((df["Marital_Status"].isin(["Married", "Divorced", "Unknown"])) & (df['Dependent_count'] >= 3)).astype(int)
df["Credit_builder_criteria"] = (df['Credit_Limit'] < 2500).astype(int)  # This threshold is chosen to include individuals with credit limits in the lower percentiles of the distribution, which may indicate a need for credit-building strategies or entry-level credit products.
df["Digital_criteria"] = (df['Contacts_Count_12_mon'] == 0).astype(int)
df["High_net_worth_individual"] = ((df['Income_Category'] == '$120K +') & (df['Total_Trans_Amt'] > 5000)).astype("Int64")
df["Rewards_maximizer"] = ((df['Total_Trans_Amt'] > 10000) & (df['Total_Revolving_Bal'] == 0)).astype(int) # For the Rewards_maximizer column, the threshold for Total_Trans_Amt is also set at $10000. Since rewards maximizers are individuals who strategically maximize rewards and benefits from credit card usage, it's reasonable to expect that they engage in higher levels of spending. Therefore, the threshold of $10000 for Total_Trans_Amt appears appropriate for identifying rewards maximizers, considering that it captures individuals with relatively high spending habits.
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

df = one_hot_encoder(df, ["Gender"], drop_first=True) # M'ler 1.
df.rename(columns={"Gender_M": "Gender"}, inplace=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.isnull().sum()
df.shape # (10102, 47)

#knn eski
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = [col for col in df.columns if col not in numeric_columns]
df_numeric = df[numeric_columns]
imputer = KNNImputer(n_neighbors=10)
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_columns)
df = pd.concat([df_numeric_imputed, df[categorical_columns]], axis=1)
df["Education_Level"] = df["Education_Level"].round().astype("Int64")
df["Income_Category"] = df["Income_Category"].round().astype("Int64")


df.isnull().sum()
df.shape
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# One-hot encoding:
df = one_hot_encoder(df, ["Marital_Status",
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

# Gizemin yarattığı ve belki onehot'a girecek kolonlar:
# 'Year_on_book', "RFM_SCORE", Segment, Cluster, RFMSegment, cluster_no

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



"""rfm skorları ile segmentasyon oluşturma"""
# Total_Trans_Amt = Monetary
# Total_Trans_Ct = Frequency
# Days_Inactive_Last_Year  Recency

# Recency: A recent purchase indicates that the customer is active and potentially more receptive to further
# communication or offers.

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
        r'5[4-5]': 'Champions'
}

# segment oluşturma (Recency + Frequency)
df['Segment'] = df['RecencyScore'].astype(str) + df['FrequencyScore'].astype(str)
df['Segment'] = df['Segment'].replace(seg_map, regex=True)
df.head(40)


# Feature scaling (robust):
# TODO GBM için scale etmeden deneyeceğiz.
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df[["Days_Inactive_Last_Year"]] = rs.fit_transform(df[["Days_Inactive_Last_Year"]])


from sklearn.cluster import KMeans
# model fit edildi.
kmeans = KMeans(n_clusters=4, max_iter=50, random_state=1)
kmeans.fit(df[['Days_Inactive_Last_Year','Total_Trans_Ct', 'Total_Trans_Amt']])

df["cluster_no"] = kmeans.labels_
df["cluster_no"] = df["cluster_no"] + 1
df.groupby("cluster_no")["Segment"].value_counts()

ssd = []

K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=1).fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])
    ssd.append(kmeans.inertia_) #inertia her bir k değeri için ssd değerini bulur.



# Optimum küme sayısını belirleme
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans(random_state=1)
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])
elbow.show()
elbow.elbow_value_

# yeni optimum kümse sayısı ile model fit edilmiştir.
kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=1).fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])


# Cluster_no 0'dan başlamaktadır. Bunun için 1 eklenmiştir.
df["cluster_no"] = kmeans.labels_
df["cluster_no"] = df["cluster_no"] + 1

df.groupby("cluster_no")["Segment"].value_counts()




cat_cols, num_cols, cat_but_car = grab_col_names(df)

df[num_cols].head()
#33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333



# Korelasyon Heatmap:
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (12, 12)})
        sns.heatmap(corr, cmap='RdBu', annot= True, vmin=-1, vmax=1)
        plt.show()
    return drop_list


drop_list = high_correlated_cols(df, plot=True)

df.drop(columns=drop_list, inplace=True, axis=1)

df.shape
#(10007, 96)

cat_cols, num_cols, cat_but_car = grab_col_names(df)



df[['MonetaryScore', 'FrequencyScore']] = df[['MonetaryScore', 'FrequencyScore']].astype(int)

dff = df.copy()
df.head()
y = df["Target"]
X = df.drop(["Target", "Segment"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Undersampling (Tomek Links)


# summarize class distribution
counter = Counter(y)
print(counter) # {0: 8489, 1: 1613} eda: ({0: 8397, 1: 1610})
# define the undersampling method
undersample = TomekLinks()
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter) # {0: 8347, 1: 1613} eda: ({0: 8247, 1: 1610})

#
#
# from imblearn.under_sampling import RandomUnderSampler
# # summarize class distribution
# print(Counter(y)) # {0: 8489, 1: 1613}
# # define undersample strategy
# undersample = RandomUnderSampler(sampling_strategy='majority')
# # fit and apply the transform
# X_over, y_over = undersample.fit_resample(X, y)
# # summarize class distribution
# print(Counter(y_over)) # {0: 1613, 1: 1613}

def model_metrics(X_train, y_train, X_test, y_test):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        model = classifier.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Classification Report
        report = classification_report(y_test, y_pred)
        print(f"Classification Report for {name}:")
        print(report)

model_metrics(X_train, y_train, X_test, y_test)


# Model:

#
# Base Models....
# Classification Report for LR:
#               precision    recall  f1-score   support
#            0       0.95      0.97      0.96      1705
#            1       0.84      0.72      0.78       316
#     accuracy                           0.94      2021
#    macro avg       0.90      0.85      0.87      2021
# weighted avg       0.93      0.94      0.93      2021
# Classification Report for KNN:
#               precision    recall  f1-score   support
#            0       0.94      0.97      0.96      1705
#            1       0.83      0.65      0.73       316
#     accuracy                           0.92      2021
#    macro avg       0.88      0.81      0.84      2021
# weighted avg       0.92      0.92      0.92      2021
# Classification Report for SVC:
#               precision    recall  f1-score   support
#            0       0.95      0.98      0.97      1705
#            1       0.87      0.73      0.79       316
#     accuracy                           0.94      2021
#    macro avg       0.91      0.85      0.88      2021
# weighted avg       0.94      0.94      0.94      2021
# Classification Report for CART:
#               precision    recall  f1-score   support
#            0       0.95      0.96      0.96      1705
#            1       0.77      0.76      0.76       316
#     accuracy                           0.93      2021
#    macro avg       0.86      0.86      0.86      2021
# weighted avg       0.93      0.93      0.93      2021
# Classification Report for RF:
#               precision    recall  f1-score   support
#            0       0.96      0.99      0.97      1705
#            1       0.92      0.76      0.83       316
#     accuracy                           0.95      2021
#    macro avg       0.94      0.87      0.90      2021
# weighted avg       0.95      0.95      0.95      2021
# Classification Report for Adaboost:
#               precision    recall  f1-score   support
#            0       0.96      0.98      0.97      1705
#            1       0.88      0.80      0.84       316
#     accuracy                           0.95      2021
#    macro avg       0.92      0.89      0.91      2021
# weighted avg       0.95      0.95      0.95      2021
# Classification Report for GBM:
#               precision    recall  f1-score   support
#            0       0.96      0.99      0.98      1705
#            1       0.93      0.79      0.86       316
#     accuracy                           0.96      2021
#    macro avg       0.95      0.89      0.92      2021
# weighted avg       0.96      0.96      0.96      2021
# Classification Report for XGBoost:
#               precision    recall  f1-score   support
#            0       0.97      0.99      0.98      1705
#            1       0.94      0.84      0.89       316
#     accuracy                           0.97      2021
#    macro avg       0.96      0.92      0.94      2021
# weighted avg       0.97      0.97      0.97      2021
# [LightGBM] [Info] Number of positive: 1297, number of negative: 6784
# [LightGBM] [Info] Total Bins 2292
# [LightGBM] [Info] Number of data points in the train set: 8081, number of used features: 94
# [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.160500 -> initscore=-1.654513
# [LightGBM] [Info] Start training from score -1.654513
# Classification Report for LightGBM:
#               precision    recall  f1-score   support
#            0       0.97      0.99      0.98      1705
#            1       0.95      0.84      0.89       316
#     accuracy                           0.97      2021
#    macro avg       0.96      0.91      0.93      2021
# weighted avg       0.97      0.97      0.97      2021
# Classification Report for CatBoost:
#               precision    recall  f1-score   support
#            0       0.97      0.99      0.98      1705
#            1       0.95      0.83      0.89       316
#     accuracy                           0.97      2021
#    macro avg       0.96      0.91      0.93      2021
# weighted avg       0.97      0.97      0.97      2021


################################################################################
svc_params = {
    'C': [1, 10],
    'gamma': [0.01, 0.1, "scale"],
    'kernel': ['rbf']
}
adaboost_params = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 0.5],
    'algorithm': ['SAMME.R']
}
knn_params = {
    "n_neighbors": range(5, 20, 5),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto'],
    'p': [1, 2]
}
cart_params = {
    'max_depth': [5, 10],
    "min_samples_split": range(10, 20, 5)
}
gbm_params = {
    "learning_rate": [0.1],
    "max_depth": [3, 8],
    "n_estimators": [100, 200],
    "subsample": [0.7]
}
rf_params = {
    "max_depth": [None, 10],
    "max_features": ["sqrt"],
    "min_samples_split": [2, 10],
    "n_estimators": [100, 200]
}
xgboost_params = {
    "learning_rate": [0.1],
    "max_depth": [3, 7],
    "n_estimators": [100, 200],
    "colsample_bytree": [0.7]
}
lightgbm_params = {
    "learning_rate": [0.1],
    "n_estimators": [100, 200],
    "max_depth": [3, 7],
    "colsample_bytree": [0.7]
}
catboost_params = {
    "learning_rate": [0.1],
    "depth": [6],
    "iterations": [100, 200],
    "subsample": [0.75]
}
logistic_params = {
    'penalty': ['l2'],
    'C': [1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200],
    'class_weight': [None, 'balanced']
}

# xgboost_params = {"learning_rate": [0.01, 0.05, 0.1, 0.5],
#                   "max_depth": [3, 5, 7, 10],
#                   "n_estimators": [50, 100, 200, 300],
#                   "colsample_bytree": [0.7, 1]}
#
# xgboost_params = {
#     "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],  # Geniş aralık, farklı öğrenme hızlarını keşfetmek için
#     "max_depth": [3, 5, 7, 10, 12],                  # Hem düşük hem de yüksek derinlikler dahil
#     "n_estimators": [50, 100, 200, 300, 400],        # Geniş aralık, daha fazla model karmaşıklığı varyasyonu için
#     "colsample_bytree": [0.5, 0.7, 0.9, 1],          # Farklı özellik alt küme oranları
#     "subsample": [0.6, 0.7, 0.8, 0.9, 1]             # Örnek alt küme oranları, çeşitliliği artırmak için
# }





##############################################################################
classifiers = [#('Adaboost', AdaBoostClassifier(), adaboost_params),
    #('KNN', KNeighborsClassifier(), knn_params),
    #("CART", DecisionTreeClassifier(), cart_params),
    #("RF", RandomForestClassifier(), rf_params),
    #("LogisticRegression", LogisticRegression(), logistic_params),  # Lojistik Regresyon
    #("SVC", SVC(), svc_params),  # Destek Vektör Makineleri
    #("GBM", GradientBoostingClassifier(), gbm_params),  # Gradyan Arttırma Makineleri
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
    #('LightGBM', LGBMClassifier(force_col_wise=True), lightgbm_params),
    #('CatBoost', CatBoostClassifier(verbose=False), catboost_params)
]


def hyperparameter_optimization(X_train, y_train, X_test, y_test, cv=5, scoring="recall"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X_train, y_train)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}")

        # Test verileri üzerinde modelin performansını değerlendir
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(f"{name} classification report:\n{report}\n")

        best_models[name] = final_model

    return best_models, gs_best.best_params_

model, best_params = hyperparameter_optimization(X_train, y_train, X_test, y_test)
# Hyperparameter Optimization....
# ########## Adaboost ##########
# roc_auc (Before): 0.9839
# roc_auc (After): 0.9851
# Adaboost best params: {'algorithm': 'SAMME.R', 'learning_rate': 0.5, 'n_estimators': 200}
# Adaboost classification report:
#               precision    recall  f1-score   support
#            0       0.97      0.98      0.98      1705
#            1       0.91      0.82      0.86       316
#     accuracy                           0.96      2021
#    macro avg       0.94      0.90      0.92      2021
# weighted avg       0.96      0.96      0.96      2021

# ########## KNN ##########
# roc_auc (Before): 0.9393
# roc_auc (After): 0.9537
# KNN best params: {'algorithm': 'auto', 'n_neighbors': 8, 'p': 1, 'weights': 'distance'}
# KNN classification report:
#               precision    recall  f1-score   support
#            0       0.94      0.98      0.96      1705
#            1       0.85      0.66      0.74       316
#     accuracy                           0.93      2021
#    macro avg       0.89      0.82      0.85      2021
# weighted avg       0.93      0.93      0.92      2021
# ########## CART ##########
# roc_auc (Before): 0.8726
# roc_auc (After): 0.9197
# CART best params: {'max_depth': 8, 'min_samples_split': 10}
# CART classification report:
#               precision    recall  f1-score   support
#            0       0.95      0.97      0.96      1705
#            1       0.83      0.74      0.78       316
#     accuracy                           0.94      2021
#    macro avg       0.89      0.86      0.87      2021
# weighted avg       0.93      0.94      0.93      2021

# ########## RF ##########
# roc_auc (Before): 0.9874
# roc_auc (After): 0.9875
# RF best params: {'max_depth': None, 'max_features': 'auto', 'min_samples_split': 2, 'n_estimators': 200}
# RF classification report:
#               precision    recall  f1-score   support
#            0       0.96      0.99      0.97      1705
#            1       0.92      0.76      0.83       316
#     accuracy                           0.95      2021
#    macro avg       0.94      0.87      0.90      2021
# weighted avg       0.95      0.95      0.95      2021
# ########## LogisticRegression ##########
# roc_auc (Before): 0.9744
# roc_auc (After): 0.9743
# LogisticRegression best params: {'C': 1, 'class_weight': None, 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga'}
# LogisticRegression classification report:
#               precision    recall  f1-score   support
#            0       0.95      0.97      0.96      1705
#            1       0.84      0.72      0.78       316
#     accuracy                           0.94      2021
#    macro avg       0.90      0.85      0.87      2021
# weighted avg       0.93      0.94      0.93      2021

# ########## SVC ##########
# roc_auc (Before): 0.9787
# roc_auc (After): 0.983
# SVC best params: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
# SVC classification report:
#               precision    recall  f1-score   support
#            0       0.96      0.97      0.97      1705
#            1       0.85      0.78      0.82       316
#     accuracy                           0.94      2021
#    macro avg       0.90      0.88      0.89      2021
# weighted avg       0.94      0.94      0.94      2021
# ########## GBM ##########
# roc_auc (Before): 0.9881

########## GBM ##########
# roc_auc (Before): 0.9881
# roc_auc (After): 0.9926
# GBM best params: {'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 1000, 'subsample': 0.5}
# GBM classification report:
#               precision    recall  f1-score   support
#            0       0.97      0.99      0.98      1705
#            1       0.95      0.81      0.88       316
#     accuracy                           0.96      2021
#    macro avg       0.96      0.90      0.93      2021
# weighted avg       0.96      0.96      0.96      2021

# ########## XGBoost ##########
# roc_auc (Before): 0.9916
# roc_auc (After): 0.9925
# XGBoost best params: {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300}
# XGBoost classification report:
#               precision    recall  f1-score   support
#            0       0.97      0.99      0.98      1705
#            1       0.94      0.85      0.90       316
#     accuracy                           0.97      2021
#    macro avg       0.96      0.92      0.94      2021
# weighted avg       0.97      0.97      0.97      2021


# LightGBM classification report:
#               precision    recall  f1-score   support
#            0       0.97      0.99      0.98      1705
#            1       0.94      0.83      0.88       316
#     accuracy                           0.96      2021
#    macro avg       0.95      0.91      0.93      2021
# weighted avg       0.96      0.96      0.96      2021

# ########## CatBoost ##########
# roc_auc (Before): 0.9935
# roc_auc (After): 0.9928
# CatBoost best params: {'depth': 6, 'iterations': 300, 'learning_rate': 0.1, 'subsample': 1.0}
# CatBoost classification report:
#               precision    recall  f1-score   support
#            0       0.97      0.99      0.98      1705
#            1       0.96      0.83      0.89       316
#     accuracy                           0.97      2021
#    macro avg       0.96      0.91      0.93      2021
# weighted avg       0.97      0.97      0.97      2021


# XCBoost Final modelini oluşturun
final_model = XGBClassifier(colsample_bytree=0.7, learning_rate=0.1, max_depth=7, n_estimators=200)

# Modeli eğitin
final_model.fit(X_train, y_train)

# Train verisi için tahmin olasılıklarını alın
train_proba = final_model.predict_proba(X_train)[:, 1]

# Test verisi için tahmin olasılıklarını alın
test_proba = final_model.predict_proba(X_test)[:, 1]

# Train verisi için ROC eğrisini hesaplayın
train_fpr, train_tpr, _ = roc_curve(y_train, train_proba)
train_auc = auc(train_fpr, train_tpr)

# Test verisi için ROC eğrisini hesaplayın
test_fpr, test_tpr, _ = roc_curve(y_test, test_proba)
test_auc = auc(test_fpr, test_tpr)

# Eğitim ve test ROC eğrilerini çiz
plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, color='blue', lw=2, label=f'Eğitim ROC (AUC = {train_auc:.2f})')
plt.plot(test_fpr, test_tpr, color='green', lw=2, label=f'Test ROC (AUC = {test_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.title('Eğitim ve Test ROC Eğrileri')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#presicion-recall eğrisi:


precision, recall, _ = precision_recall_curve(y_test, test_proba)

# Precision-Recall eğrisini çiz
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Eğrisi')
plt.grid(True)
plt.show()

# Detaylı sınıflandırma raporunu alın
y_pred = final_model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['Negatif', 'Pozitif'])
print(report)


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score



################################
# Analyzing Model Complexity with Learning Curves (BONUS)
################################
#
# def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
#     train_score, test_score = validation_curve(
#         model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)
#
#     mean_train_score = np.mean(train_score, axis=1)
#     mean_test_score = np.mean(test_score, axis=1)
#
#     plt.plot(param_range, mean_train_score,
#              label="Training Score", color='b')
#
#     plt.plot(param_range, mean_test_score,
#              label="Validation Score", color='g')
#
#     plt.title(f"Validation Curve for {type(model).__name__}")
#     plt.xlabel(f"Number of {param_name}")
#     plt.ylabel(f"{scoring}")
#     plt.tight_layout()
#     plt.legend(loc='best')
#     plt.show(block=True)
#
#
# for param_name, param_range in gbm_params.items():
#     val_curve_params(final_model, X, y, param_name, param_range)




################################
# Feature Importance
################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(final_model, X)


#top 15
plot_importance(final_model, X, num=15)

# # GBM
# def get_top_features(model, features, num=15):
#     feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
#     top_features = feature_imp.sort_values(by="Value", ascending=False).head(num)['Feature'].tolist()
#     return top_features
#
# top_15_features = get_top_features(final_model, X)
# print(top_15_features)
#
# selected_columns = top_15_features + ["Target"]
# df_gbm = df[selected_columns]
#
#
# y = df_gbm["Target"]
# X = df_gbm.drop(["Target"], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # GBM modelini oluşturun
# final_model = GradientBoostingClassifier(learning_rate = 0.1, max_depth= 3, n_estimators= 1000, subsample= 0.7)
#
# # Modeli eğitin
# final_model.fit(X_train, y_train)
#
# # Detaylı sınıflandırma raporunu alın
# y_pred = final_model.predict(X_test)
# report = classification_report(y_test, y_pred, target_names=['Negatif', 'Pozitif'])
# print(report)
# #               precision    recall  f1-score   support
# #      Negatif       0.97      0.99      0.98      1705
# #      Pozitif       0.94      0.83      0.88       316
# #     accuracy                           0.97      2021
# #    macro avg       0.96      0.91      0.93      2021
# # weighted avg       0.96      0.97      0.96      2021


# KNN
def get_top_features(model, features, num=15):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    top_features = feature_imp.sort_values(by="Value", ascending=False).head(num)['Feature'].tolist()
    return top_features

top_15_features = get_top_features(final_model, X)
print(top_15_features)

selected_columns = top_15_features + ["Target"]
df_gbm = df[selected_columns]


y = df_gbm["Target"]
X = df_gbm.drop(["Target"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GBM modelini oluşturun
final_model = KNeighborsClassifier()(learning_rate = 0.1, max_depth= 3, n_estimators= 1000, subsample= 0.7)

# Modeli eğitin
final_model.fit(X_train, y_train)

# Detaylı sınıflandırma raporunu alın
y_pred = final_model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['Negatif', 'Pozitif'])
print(report)
#               precision    recall  f1-score   support
#      Negatif       0.97      0.99      0.98      1705
#      Pozitif       0.94      0.83      0.88       316
#     accuracy                           0.97      2021
#    macro avg       0.96      0.91      0.93      2021
# weighted avg       0.96      0.97      0.96      2021


# burada error veriyor category'ye çevirmek
import shap
# SHAP değerlerini hesaplamak için CatBoost açıklayıcısını kullanın
explainer = shap.TreeExplainer(final_model, cat_features="cluster_no")
shap_values = explainer.shap_values(X)

# SHAP özet grafiği
shap.summary_plot(shap_values, X, plot_type="bar")




