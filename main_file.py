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

# Fonksiyonlarımız:
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

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
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

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

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

################ CHATGPT BAŞLANGIÇ

from sklearn.neighbors import LocalOutlierFactor

dff = df.copy()
def remove_outliers_iqr(df, col):
    q1 = df[col].quantile(0.05)
    q3 = df[col].quantile(0.95)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
    return outliers

def remove_outliers_lof(df, num_cols):
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(df[num_cols])

    df_scores = clf.negative_outlier_factor_

    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked=True, xlim=[0, 100], style='.-')
    plt.show()

    return df_scores

th = np.sort(df_scores)[25]
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Remove outliers using IQR
iqr_outliers = []
for col in num_cols:
    outliers = remove_outliers_iqr(df, col)
    iqr_outliers.extend(outliers)

# Replace outliers with thresholds using IQR
for col in num_cols:
    outliers = remove_outliers_iqr(df, col)
    df.loc[outliers, col] = df[col].median()  # Replace outliers with median

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Remove outliers using LOF
lof_outliers = remove_outliers_lof(df, num_cols)

# Plot LOF scores
clf = LocalOutlierFactor(n_neighbors=20)
df_scores = clf.negative_outlier_factor_


# Remove outliers detected by LOF
lof_outliers = df[df_scores == -1].index
df_cleaned = df.drop(lof_outliers)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Sadece IQR:
len(iqr_outliers) # 40
# [2, 4, 7, 8, 12, 46, 47, 58, 154, 219, 284, 466, 658, 773, 841, 1219, 1, 2, 3, 4, 12, 30, 68, 91, 113, 131, 146, 158, 162, 167, 190, 239, 269, 280, 366, 757, 773, 805, 1095, 2510]

# Sadece lof:
len(lof_outliers) # 101

aynı_indexler = [item for item in iqr_outliers if item in lof_outliers]
len(aynı_indexler) # sadece 158. index var.

### şimdi sırayla yapalım. önce iqr sonra lof
len(lof_outliers) # 101

### şimdi önce lof sonra iqr
len(iqr_outliers) # 40
# [2, 4, 7, 8, 12, 46, 47, 58, 154, 219, 284, 466, 658, 773, 841, 1219, 1, 2, 3, 4, 12, 30, 68, 91, 113, 131, 146, 158, 162, 167, 190, 239, 269, 280, 366, 757, 773, 805, 1095, 2510]



# Output the index values of outliers removed by each method
print("Outliers removed by IQR method:")
print(iqr_outliers)
print("\nOutliers removed by LOF method:")
print(lof_outliers)


################ CHATGPT BİTİŞ

# Outlier temizleme (IQR ve LOF):
# IQR
for col in num_cols:
    print(col)
    grab_outliers(df, col)

for col in num_cols:
    replace_with_thresholds(df, col)

# LOF
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols])

df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 100], style='.-')
plt.show()

th = np.sort(df_scores)[25]

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# Missing values
cols_with_unknown = ['Income_Category', "Education_Level"]
for col in cols_with_unknown:
    df[col] = df[col].apply(lambda x: np.nan if x == 'Unknown' else x)

# ANALİZ BAŞLANGICI
df.groupby("Income_Category")["Card_Category"].value_counts()
df.groupby("Card_Category")["Credit_Limit"].mean()
df.groupby("Card_Category")["Target"].mean()
df["Total_Relationship_Count"].value_counts()
df.groupby("Total_Relationship_Count")["Target"].mean()
# 1   0.256
# 2   0.278
# 3   0.174
# 4   0.118
# 5   0.120
# 6   0.105
# TODO churn etmesi beklenen müşteriye, bankanın başka ürünlerinden kampanyalı satış yapmaya çalışmalıyız.

df.groupby("Months_Inactive_12_mon")["Target"].mean()
# 0   0.517
# 1   0.045
# 2   0.154
# 3   0.215
# 4   0.299
# 5   0.180
# 6   0.153
df.groupby("Months_on_book")["Months_Inactive_12_mon"].mean()


df.groupby("Target")["Months_on_book"].mean()
df["Months_on_book"].value_counts()
df["Months_on_book"].describe().T
# TODO bu aşağıdakinden Gizem de yapmış, pushlayınca onunkini alırız.
df["On_book_cat"] = np.where((df["Months_on_book"] < 12), "<1_year", np.where((df["Months_on_book"] < 24), "<2_years", np.where((df["Months_on_book"] < 36), "<3_years", np.where((df["Months_on_book"] < 48), "<4_years", "<5_years"))))
df["On_book_cat"].value_counts()
df.groupby("On_book_cat")["Target"].mean() # Anlamlı değil


df.groupby("Months_Inactive_12_mon")["Total_Trans_Amt"].mean()
df.groupby("Months_Inactive_12_mon")["Total_Trans_Ct"].mean()
df["Total_Trans_Ct"].describe().T
df.loc[df["Total_Trans_Ct"] == 139]
df["Total_Trans_Amt"].describe().T
df.loc[df["Total_Trans_Amt"] == 510]

df.loc[(df['Total_Revolving_Bal'] > 2500)].count()
(df.loc[(df['Total_Revolving_Bal'] > 2500)])["Target"].mean()
df.loc[df['Total_Revolving_Bal'] > 2510].value_counts()
df["Total_Revolving_Bal"].describe().T


df.groupby("Target")["Avg_Utilization_Ratio"].mean() # TODO borcu düşük olanların churn etme oranı daha yüksek (x3.5).
# 0   0.296
# 1   0.162

df.groupby("Target")["Total_Revolving_Bal"].mean() # TODO
# 0   1256.604
# 1    672.823


df.groupby("Income_Category")["Total_Revolving_Bal"].mean()
df.groupby("Income_Category")["Total_Trans_Amt"].mean()
# TODO 1. Çekilen kredi miktarı da, yapılan harcama miktarı da müşteri gelirlerine kıyasla stabil.
# TODO 2. Bu da, düşük gelirli müşterilerin, Avg_Utilization_Ratio'sunun yani borç ödeme zorluğu oranını artırıyor.
# TODO 3. Borcu olan müşteriler, bankadan ayrılamıyor.
# TODO 4. Müşterileri bankada tutmak için A) ürün sat, B) borcunu artır -- mesela kk limitini artırmayı teklif et.

df['Total_Amt_Increased'] = np.where((df['Total_Amt_Chng_Q4_Q1'] > 0) & (df['Total_Amt_Chng_Q4_Q1'] < 1), 0, 1)

df["Has_debt"] = np.where((df["Credit_Limit"] > df["Avg_Open_To_Buy"]), 1, 0).astype(int)
#df["Credit_increased"].value_counts()
#df["Avg_Utilization_Ratio"].describe().T
#df.loc[df["Avg_Utilization_Ratio"] == 0].head()

df.head()

# TODO Şirket mottosu: "Biz borçlunun yanındayız!"

df["Important_client_score"] = df["Total_Relationship_Count"] * (df["Months_on_book"] / 12)
df["Important_client_score"].describe().T
num_summary(df, "Important_client_score", plot=True)

df.groupby("Target")["Important_client_score"].mean()
# 0   11.701
# 1    9.863
# TODO Banka, önemli müşterileri tutmakta başarılı!

df["Avg_Trans_Amt"] = df["Total_Trans_Amt"] / df['Total_Trans_Ct']

labels = ['Young', 'Middle_Aged', 'Senior']
bins = [25, 35, 55, 74]
df['Customer_Age_Category'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels)

df["Days_Inactive_Last_Year"] = df["Months_Inactive_12_mon"] * 30

df = df.sort_values(by="Days_Inactive_Last_Year", ascending=True)
df.reset_index(drop=True, inplace=True)


# Yeni bir "Recency" sütunu oluştur
df['RecencyScore'] = np.nan

# İlk 2025 satırı 5 olarak ayarla
df.loc[:2024, 'RecencyScore'] = 5

# Sonraki 2025 satırı 4 olarak ayarla
df.loc[2025:4049, 'RecencyScore'] = 4

# Sonraki 2027 satırı 3 olarak ayarla
df.loc[4050:6076, 'RecencyScore'] = 3

# Sonraki 2025 satırı 2 olarak ayarla
df.loc[6077:8101, 'RecencyScore'] = 2

# Kalan 2025 satırı 1 olarak ayarla
df.loc[8102:, 'RecencyScore'] = 1


df["MonetaryScore"] = pd.qcut(df["Total_Trans_Amt"], 5, labels=[1, 2, 3, 4, 5])
df["FrequencyScore"] = pd.qcut(df["Total_Trans_Ct"], 5, labels=[1, 2, 3, 4, 5])
# # Total_Trans_Amt = Monetary
# # Total_Trans_Ct = Frequency
# # Days_Inactive_12_mon = Recency

combine_categories(df, 'Customer_Age_Category', 'Marital_Status', 'Age_&_Marital')
combine_categories(df, 'Gender', 'Customer_Age_Category', 'Gender_&_Age')
combine_categories(df, "Card_Category", "Customer_Age_Category", "Card_&_Age")
combine_categories(df, "Gender", "FrequencyScore", "Gender_&_Frequency")
combine_categories(df, "Gender", "MonetaryScore", "Gender_&_Monetary")

df['Total_Amt_Increased'] = np.where((df['Total_Amt_Chng_Q4_Q1'] >= 0) & (df['Total_Amt_Chng_Q4_Q1'] < 1), 0, 1)
df['Total_Ct_Increased'] = np.where((df['Total_Ct_Chng_Q4_Q1'] >= 0) & (df['Total_Ct_Chng_Q4_Q1'] < 1), 0, 1)
df['Total_Ct_Chng_Q4_Q1'].describe().T
df['Total_Amt_Chng_Q4_Q1'].describe().T
df.loc[df['Total_Amt_Chng_Q4_Q1'] == 0]

# Total_Ct_Chng_Q4_Q1= Q4/Q1 olduğuna göre, bunun 0 olduğu yerlerde Q4 = 0, yani recency'si 3 ay olur.
df.loc[df["Total_Ct_Chng_Q4_Q1"]==0]

# İşlem sayısı ve miktarı pattern'leri:
# İşlem sayısı aynı kalıp, harcama miktarı artanlar: (belki daha çok para kazanmaya başlamışlardır)(TODO kredi limiti ile incele)
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
# İşlem sayısı azalmış, miktar da azalmış. Churn eder mi acaba?
df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Dec_ct_dec_amt"
# (df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1)])["Target"].mean() # 0.17
df.head()

df.groupby("Ct_vs_Amt")["Target"].mean()
# Count arttıkça churn etme olasılığı azalıyor.
df.groupby("Target")["Total_Trans_Ct"].mean()


df["Contacts_Count_12_mon"].describe().T
df.groupby("Contacts_Count_12_mon")["Target"].mean() # 6'ların hepsi churn. Yükseldikçe churn olasılığı artıyor.
# TODO Number of contacts with the bank might indicate dissatisfaction or queries.
# 0   0.018
# 1   0.072
# 2   0.125
# 3   0.201
# 4   0.226
# 5   0.335
# 6   1.000


# Personalar
df["Affluent_criteria"] = (df['Income_Category'] == '$120K +').astype(int)
df["Budget_criteria"] = ((df['Income_Category'] == 'Less than $40K') & (df['Education_Level'].isin(['High School', 'College']))).astype(int)
df["Young_prof_criteria"] = ((df['Customer_Age'] <= 30) & (df['Education_Level'].isin(['College', 'Graduate']))).astype(int)
df["Family_criteria"] = ((df["Marital_Status"].isin(["Married", "Divorced", "Unknown"])) & (df['Dependent_count'] >= 3)).astype(int)
df["Credit_builder_criteria"] = (df['Credit_Limit'] < 2500).astype(int)  # This threshold is chosen to include individuals with credit limits in the lower percentiles of the distribution, which may indicate a need for credit-building strategies or entry-level credit products.
df["Digital_criteria"] = (df['Contacts_Count_12_mon'] == 0).astype(int)
df["High_net_worth_individual"] = ((df['Income_Category'] == '$120K +') & (df['Total_Trans_Amt'] > 5000)).astype(int)
df["Rewards_maximizer"] = ((df['Total_Trans_Amt'] > 10000) & (df['Total_Revolving_Bal'] == 0)).astype(int) # For the Rewards_maximizer column, the threshold for Total_Trans_Amt is also set at $10000. Since rewards maximizers are individuals who strategically maximize rewards and benefits from credit card usage, it's reasonable to expect that they engage in higher levels of spending. Therefore, the threshold of $10000 for Total_Trans_Amt appears appropriate for identifying rewards maximizers, considering that it captures individuals with relatively high spending habits.
df["May_marry"] = ((df["Age_&_Marital"] == "Young_Single") & (df['Dependent_count'] == 0)).astype(int)

# Total_Trans_Amt threshold'larını inceleyip üsttekiler için ayarlama yapalım (üsttekiler ayarlama yapılmış hali):
(df.loc[df['Total_Trans_Amt'] > 10000]).groupby("Income_Category")["Customer_Age"].mean()
(df.loc[df['Total_Trans_Amt'] > 10000]).groupby("Income_Category").count()
df['Total_Trans_Amt'].describe().T

df.head()

# TODO öneri: Total_dependent_count fazla olanlara ek kart öner.

df.groupby("Income_Category")["Avg_Open_To_Buy"].mean()
df.groupby("Income_Category")["Credit_Limit"].mean()

df["Product_by_Year"] = df["Total_Relationship_Count"] / (df["Months_on_book"] / 12)
df["Product_by_Year"].describe().T
num_summary(df, "Product_by_Year", plot=True)


df.head(20)
df.shape
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.info()


# Credit limit - total revolvinng bal = avg open to buy
# df[["Credit_Limit","Total_Revolving_Bal","Avg_Open_To_Buy"]].head(20)
#
# df["fark"] = df["Credit_Limit"] - df["Total_Revolving_Bal"]
#
# df[["fark", "Avg_Open_To_Buy"]].head(50)

# total revolving bal / credit limit = avg_utilication_ratio
# df[["Total_Revolving_Bal","Credit_Limit","Avg_Utilization_Ratio"]].head(20)
#
# df["bölüm"] = df["Total_Revolving_Bal"] / df["Credit_Limit"]
#
# df[["bölüm", "Avg_Utilization_Ratio"]].head(50)


# Scatter plot çizimi
plt.figure(figsize=(10, 10))
sns.scatterplot(x='Credit_Limit', y='Total_Revolving_Bal', hue='Income_Category', data=df, s=20)
plt.xlabel('Credit Limit')
plt.ylabel('Total Revolving Balance')
plt.title('Scatter Plot of Total Revolving Balance vs. Credit Limit by Income Category')
plt.tight_layout()
plt.show()


# Müşterinin yaşını ve bankada geçirdiği süreyi birleştirerek uzun süreli müşteri olup olmadığını gösteren bir değişken oluşturma
# Ay bilgilerini yıla çevirerek yeni bir sütun oluşturma
df['Year_on_book'] = df['Months_on_book'] // 12
df['Year_on_book'].value_counts()
# Year_on_book
# 3    5508
# 2    3115
# 4     817
# 1     687

"""rfm skorları ile segmentasyon oluşturma"""

df['RecencyScore'] = df['RecencyScore'].astype(int)

# rfm score oluşturma
df["RFM_SCORE"] = df['RecencyScore'].astype(str) + df['FrequencyScore'].astype(str) + df['MonetaryScore'].astype(str)

df[["RFM_SCORE", "RecencyScore","FrequencyScore", "MonetaryScore" ]].head()

seg_map = {
        r'[1-2][1-2]': 'Hibernating',
        r'[1-2][3-4]': 'At Risk',
        r'[1-2]5': 'Can\'t Loose',
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
df['Segment'].head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# k-means ile müşteri segmentasyonu öncesi standartlaştırmayı yapmak gerek
# Min-Max ölçeklendirme
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler((0,1))
df[['Total_Trans_Amt','Total_Trans_Ct','Days_Inactive_Last_Year']] = sc.fit_transform(df[['Total_Trans_Amt','Total_Trans_Ct','Days_Inactive_Last_Year']])


from sklearn.cluster import KMeans
# model fit edildi.
kmeans = KMeans(n_clusters = 10)
k_fit = kmeans.fit(df[['Days_Inactive_Last_Year','Total_Trans_Ct']])
# Total_Trans_Amt = Monetary
# Total_Trans_Ct = Frequency
# Months_Inactive_12_mon  Recency

# merkezler
centers = kmeans.cluster_centers_

segments = kmeans.labels_
df['Cluster'] = segments+1 #kümeler 0'dan başlamasın diye
df.head()


df['RFMSegment'] = np.array(df['Cluster'])

df.groupby(['Cluster','RFMSegment'])['RFMSegment'].count()



# Encoding:
dff = df.copy()

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

""" Kullanmadık ama mesela Card_&_Age'de ve Age_&_Marital'da 0.005 ratio'lu kategoriler var
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)] # any() çünkü col'un value_counts/len'ini yani value'larının yüzdelik ratio'larını alınca 0.01'den düşük herhangi biri (ANY) varsa, col'u al getir diyor.

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)     # bu ratio tablosunu, indeksi value (e.g. male/female), value'su ratio olacak şekilde pd.series (indeksli list) olarak kaydettim.
        rare_labels = tmp[tmp < rare_perc].index    # sonra bu listede değeri 0.01'den küçük olanların indexini=label'ını kaydettim.
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
        #temp_df["EMERGENCYSTATE_MODE"].isin(rare_labels) # output: tek bir sütun için her bir girdinin rare_labels'da olup olmamasına göre T/F döndürdü.

        # type(rare_columns) = pandas.series
        # tmp.dtype = float

    return temp_df


new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)
"""


# Ordinal encoding:
def ordinal_encoder(dataframe, col):
    edu_cats = ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', np.nan]
    income_cats = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', np.nan]
    customer_age_cat = ['Young', 'Middle_Aged', 'Senior']
    card_cat = ['Blue', 'Silver', 'Gold_Platinum']
    on_book_cat = ["<2_years", "<3_years", "<4_years", "<5_years"]

    if col == "Education_Level":
        col_cats = edu_cats
    if col == "Income_Category":
        col_cats = income_cats
    if col == "Customer_Age_Category":
        col_cats = customer_age_cat
    if col == "Card_Category":
        col_cats = card_cat
    if col == "On_book_cat":
        col_cats = on_book_cat

    ordinal_encoder = OrdinalEncoder(categories=[col_cats])  # burada direkt int alamıyorum çünkü NaN'lar mevcut.
    df[col] = ordinal_encoder.fit_transform(df[[col]])

    print(df[col].head(20))
    return df

for col in ["Education_Level", "Income_Category", "Customer_Age_Category", "Card_Category", "On_book_cat"]:
    df = ordinal_encoder(df, col)

df.columns
df.head()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# One-hot encoding:
df = one_hot_encoder(df, ["Gender",
                          "Marital_Status",
                          "Age_&_Marital",
                          "Gender_&_Age",
                          "Card_&_Age",
                          "Gender_&_Frequency",
                          "Gender_&_Monetary",
                          'Ct_vs_Amt',
                          'Dependent_count',
                          'Total_Relationship_Count',
                          'Months_Inactive_12_mon',
                          'Contacts_Count_12_mon',
                          'MonetaryScore',
                          'FrequencyScore'], drop_first=True)

# Gizemin yarattığı ve belki onehot'a girecek kolonlar:
# 'Year_on_book', "RFM_SCORE", Segment, Cluster, RFMSegment, cluster_no

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
# df.drop(useless_cols, axis=1, inplace=True)

df.head()
dff = df.copy()
df = dff.copy()

# Nan doldurma:
imputer = KNNImputer(n_neighbors=10)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

df["Education_Level"] = df["Education_Level"].round().astype(int)
df["Income_Category"] = df["Income_Category"].round().astype(int)


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Değişken tipi dönüştürme:
for col in df.columns:
    if df[col].dtype == 'float64':  # Sadece float sütunları kontrol edelim
        if (df[col] % 1 == 000).all():  # Tüm değerlerin virgülden sonrası 0 mı kontrol edelim
            df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Feature scaling (robust):
# TODO GBM için scale etmeden deneyeceğiz.
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

df["Cluster"].value_counts()


#liste olusturduk.
ssd = []

K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters = k).fit(df[['Days_Inactive_Last_Year','Total_Trans_Ct','Total_Trans_Amt']])
    ssd.append(kmeans.inertia_) #inertia her bir k değeri için ssd değerini bulur.

plt.plot(K, ssd, "bx-")
plt.xlabel("Distance Residual Sums Versus Different k Values")
plt.title("Elbow method for Optimum number of clusters")

from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k = (2,20))
visu.fit(df[['Days_Inactive_Last_Year','Total_Trans_Ct','Total_Trans_Amt']])
visu.poof();
# k = 6 çıktı


# Total_Trans_Amt = Monetary
# Total_Trans_Ct = Frequency
# Months_Inactive_12_mon  Recency
# yeni optimum kümse sayısı ile model fit edilmiştir.
kmeans = KMeans(n_clusters = 5).fit(df[['Days_Inactive_Last_Year','Total_Trans_Ct','Total_Trans_Amt']])
kumeler = kmeans.labels_
pd.DataFrame({"Customer ID": df.index, "Kumeler": kumeler})

# Cluster_no 0'dan başlamaktadır. Bunun için 1 eklenmiştir.
df["cluster_no"] = kumeler
df["cluster_no"] = df["cluster_no"] + 1

df.head()



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


cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Multicollinearity test:
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
    return vif_data


all_independent_variables = df[num_cols]
vif_results = calculate_vif(all_independent_variables)
print(vif_results)

# Model:
y = df["Target"]
X = df.drop(["Target"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Hiperparametre Optimizasyonu ve Model:

rf_params = {"max_depth": [5, 10, 15, None],
             #"max_features": [2, 4, 8, 16, 30],
             "min_samples_split": [2, 5, 10],
             "n_estimators": [50, 100, 200, 300]}

xgboost_params = {"learning_rate": [0.01, 0.05, 0.1, 0.5],
                  "max_depth": [3, 5, 7, 10],
                  "n_estimators": [50, 100, 200, 300]}

lightgbm_params = {"learning_rate": [0.01, 0.05, 0.1, 0.5],
                   "n_estimators": [50, 100, 200, 300],
                   "max_depth": [3, 5, 7, 10]}

gbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1],
              "n_estimators": [50, 100, 200, 300],
              "max_depth": [3, 5, 7, 10],
              "subsample": [0.5, 0.75, 1.0]}

catboost_params = {"learning_rate": [0.01, 0.05, 0.1, 0.5],
                   "depth": [3, 5, 7, 10],
                   "iterations": [50, 100, 200, 300],
                   "subsample": [0.5, 0.75, 1.0]}

adaboost_params = { "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5],
                    "base_estimator__max_depth": [1, 2, 3, 4],
                    "random_state": [None, 42]}

classifiers = ([("RF", RandomForestClassifier(), rf_params),
                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
                ('LightGBM', LGBMClassifier(force_col_wise=True), lightgbm_params),
                ('GBM', GradientBoostingClassifier(), gbm_params),
                ('CatBoost', CatBoostClassifier(verbose=False), catboost_params),
                ('AdaBoost', CatBoostClassifier(), adaboost_params),
                ])


def hyperparameter_optimization(X_train, y_train, X_test, y_test, cv=3, scoring="roc_auc"):
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

    return best_models

hyperparameter_optimization(X_train, y_train, X_test, y_test)


################################
# Analyzing Model Complexity with Learning Curves (BONUS)
################################
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]