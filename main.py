# Gerekli kütüphaneleri import etme ve console display ayarları
from catboost import CatBoostClassifier
from imblearn.under_sampling import TomekLinks
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, auc, precision_recall_curve)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer
import warnings
warnings.simplefilter(action="ignore")
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Değişkenler ve açıklamaları

# Müşteri hakkındaki kişisel bilgiler:
    # CLIENTNUM: Müşteri numarası. Tekrar eden değer yok.
    # Attrition_Flag: Hedef değişken. Binary. Müşteri churn olmuş ise 1, yoksa 0.
    # Customer_Age: Müşterinin yaşı
    # Gender: Müşterinin cinsiyeti (F, M)
    # Dependent_count: Müşterinin bakmakla yükümlü olduğu kişi sayısı (0, 1, 2, 3, 4, 5)
    # Education_Level: Müşterinin eğitim seviyesi (Uneducated, High School, College, Graduate, Post-Graduate, Doctorate, Unknown)
    # Marital_Status: Müşterinin medeni durumu (Single, Married, Divorced, Unknown)
    # Income_Category: Müşterinin hangi gelir kategorisinde olduğu bilgisi (Less than $40K, $40K - $60K, $60K - $80K, $80K - $120K, $120K+, Unknown)

    # Müşterinin bankayla ilişkisi hakkındaki bilgiler:
    # Card_Category: Müşterinin sahip olduğu kredi kartı türü (Blue, Silver, Gold, Platinum)
    # Months_on_book: Müşterinin bu bankayla çalıştığı ay sayısı
    # Total_Relationship_Count: Müşterinin bankaya ait ürünlerden kaçına sahip olduğu (1, 2, 3, 4, 5, 6)
    # Months_Inactive_12_mon: Müşterinin son 12 aylık sürede kredi kartını kullanmadığı ay sayısı
    # Contacts_Count_12_mon: Müşteriyle son 12 ayda kurulan iletişim sayısı (0, 1, 2, 3, 4, 5, 6)
    # Credit_Limit: Müşterinin kredi kartı limiti
    # Total_Revolving_Bal: Devir bakiyesi. Müşterinin ödemeyi taahhüt ettiği ancak henüz ödenmemiş olan aylık taksitli borç miktarı
    # Avg_Open_To_Buy: Müşterinin borç taahhütlerinden sonra arta kalan, harcayabileceği miktar (Credit_Limit - Total_Revolving_Bal)
    # Avg_Utilization_Ratio: Müşterinin mevcut kredi kartı borçlarının kredi limitine oranı (Total_Revolving_Bal / Credit_Limit)
    # Total_Trans_Amt: Müşterinin son 12 aydaki kredi kartı işlemlerinin tutar toplamı
    # Total_Amt_Chng_Q4_Q1: Müşterinin 4. çeyrekteki harcama tutarının, 1. çeyrekteki harcama tutarına kıyasla artış/azalış hareketini gösterir (4. Çeyrek / 1. Çeyrek)
    # Total_Trans_Ct: Müşterinin son 12 aydaki kredi kartı işlemlerinin adet toplamı
    # Total_Ct_Chng_Q4_Q1: Müşterinin 4. çeyrekteki harcama adedinin, 1. çeyrekteki harcama adedine kıyasla artış/azalış hareketini gösterir (4. Çeyrek / `1. Çeyrek)

# Fonksiyonlar
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

def remove_outliers_from_all_columns(dataframe):
    for col_name in num_cols:
        low, up = outlier_thresholds(dataframe, col_name)  # Aykırı değer sınırlarını hesapla
        outliers = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]
        print(f"{col_name} için aykırı değer sayısı: {outliers.shape[0]}")
        # Aykırı değerleri dataframe'den çıkar
        dataframe = dataframe.drop(outliers.index).reset_index(drop=True)
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

def combine_categories(df, cat_col1, cat_col2, new_col_name):
    df[new_col_name] = df[cat_col1].astype(str) + '_' + df[cat_col2].astype(str)

########################################################################################################################
# Base model
########################################################################################################################
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

df = one_hot_encoder(df, ["Gender",
                         "Education_Level",
                         "Marital_Status",
                         "Income_Category",
                         "Card_Category"],
                         drop_first=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


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


########################################################################################################################
# Feature Engineering
########################################################################################################################

# Veri setini yeniden yükleme
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
for col in num_cols:
    print(col, grab_outliers(df, col))

df = remove_outliers_from_all_columns(df)
df.shape


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

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Eksik değer analizi
cols_with_unknown = ['Income_Category', "Education_Level"]
for col in cols_with_unknown:
    df[col] = df[col].apply(lambda x: np.nan if x == 'Unknown' else x)

# Analiz Başlangıcı (Yeni değişkenler üretme ve inceleme)
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
df.loc[(df["Total_Ct_Chng_Q4_Q1"] == 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Same_ct_same_amt"
df.loc[(df["Total_Ct_Chng_Q4_Q1"] == 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Same_ct_dec_amt"
df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] > 1), "Ct_vs_Amt"] = "Inc_ct_inc_amt"
df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Inc_ct_same_amt"
df.loc[(df["Total_Ct_Chng_Q4_Q1"] > 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Inc_ct_dec_amt"
df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] > 1), "Ct_vs_Amt"] = "Dec_ct_inc_amt"
df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] == 1), "Ct_vs_Amt"] = "Dec_ct_same_amt"
df.loc[(df["Total_Ct_Chng_Q4_Q1"] < 1) & (df["Total_Amt_Chng_Q4_Q1"] < 1), "Ct_vs_Amt"] = "Dec_ct_dec_amt"


# Persona yaratma
df["Affluent_criteria"] = (df['Income_Category'] == '$120K +').astype("Int64")
df["Budget_criteria"] = ((df['Income_Category'] == 'Less than $40K') & (df['Education_Level'].isin(['High School', 'College']))).astype("Int64")
df["Young_prof_criteria"] = ((df['Customer_Age'] <= 30) & (df['Education_Level'].isin(['College', 'Graduate']))).astype("Int64")
df["Family_criteria"] = ((df["Marital_Status"].isin(["Married", "Divorced", "Unknown"])) & (df['Dependent_count'] >= 3)).astype(int)
df["Credit_builder_criteria"] = (df['Credit_Limit'] < 2500).astype(int)
df["Digital_criteria"] = (df['Contacts_Count_12_mon'] == 0).astype(int)
df["High_net_worth_individual"] = ((df['Income_Category'] == '$120K +') & (df['Total_Trans_Amt'] > 5000)).astype("Int64")
df["Rewards_maximizer"] = ((df['Total_Trans_Amt'] > 10000) & (df['Total_Revolving_Bal'] == 0)).astype(int)
df["May_marry"] = ((df["Age_&_Marital"] == "Young_Single") & (df['Dependent_count'] == 0)).astype(int)

df["Product_by_Year"] = df["Total_Relationship_Count"] / (df["Months_on_book"] / 12)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Encoding işlemleri:
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

# Ordinal encoding
def ordinal_encoder(dataframe, col):
    if col in categories_dict:
        col_cats = categories_dict[col]
        ordinal_encoder = OrdinalEncoder(categories=[col_cats])
        dataframe[col] = ordinal_encoder.fit_transform(dataframe[[col]])

    return dataframe

for col in df.columns:
    ordinal_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# One-hot encoding
df = one_hot_encoder(df, ["Gender"], drop_first=True) # M'ler 1.
df.rename(columns={"Gender_M": "Gender"}, inplace=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.isnull().sum()
df.shape

# KNN imputer ile boş değerleri doldurma
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = [col for col in df.columns if col not in numeric_columns]
df_numeric = df[numeric_columns]
imputer = KNNImputer(n_neighbors=10)
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_columns)
df = pd.concat([df_numeric_imputed, df[categorical_columns]], axis=1)
df["Education_Level"] = df["Education_Level"].round().astype("Int64")
df["Income_Category"] = df["Income_Category"].round().astype("Int64")


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# One-hot encoding
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


# Bilgi sağlamayan kolonları silme işlemi
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
df.drop(useless_cols, axis=1, inplace=True)

df.head()



cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Değişken tipi dönüştürme:
for col in df.columns:
    if df[col].dtype == 'float64':  # Sadece float sütunları kontrol edelim
        if (df[col] % 1 == 000).all():  # Tüm değerlerin virgülden sonrası 0 mı kontrol edelim
            df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)



#  RFM skorları ile müşteri segmentasyon oluşturma
# Total_Trans_Amt = Monetary
# Total_Trans_Ct = Frequency
# Days_Inactive_Last_Year = Recency


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

# Segment oluşturma (RecencyScore + FrequencyScore)
df['Segment'] = df['RecencyScore'].astype(str) + df['FrequencyScore'].astype(str)
df['Segment'] = df['Segment'].replace(seg_map, regex=True)
df.head(40)


# Feature scaling (Robust Scaler):
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df[["Days_Inactive_Last_Year"]] = rs.fit_transform(df[["Days_Inactive_Last_Year"]])


# Optimum küme sayısını belirleme
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans(random_state=1)
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])
elbow.show()
elbow.elbow_value_

# Yeni optimum kümse sayısı ile model fit etme
kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=1).fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])


# Cluster_no 1'den başlatma
df["cluster_no"] = kmeans.labels_
df["cluster_no"] = df["cluster_no"] + 1


# Yeni oluşturulan cluster'ların segment'leri referans alınarak ordinal encode edilmesi
df.groupby("cluster_no")["Segment"].value_counts()
df['cluster_no'] = df['cluster_no'].replace({1: 4, 2: 7, 3: 5, 4: 6, 5: 1, 6: 2, 7: 3})


cat_cols, num_cols, cat_but_car = grab_col_names(df)


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

df[['MonetaryScore', 'FrequencyScore']] = df[['MonetaryScore', 'FrequencyScore']].astype(int)

########################################################################################################################
# Model - Feature Engineering Sonrası
########################################################################################################################

y = df["Target"]
X = df.drop(["Target", "Segment"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Undersampling (Tomek Links)
counter = Counter(y_train)
print(counter)
undersample = TomekLinks()
X_train, y_train = undersample.fit_resample(X_train, y_train)
counter = Counter(y_train)
print(counter)


# Base Model (Özellik mühendisliği sonrası)
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

########################################################################################################################
# Model - Hiperparametre Optimizasyonu Sonrası
########################################################################################################################

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
   "learning_rate": [0.05, 0.1, 0.2],
   "max_depth": [3, 5, 7],
   "n_estimators": [100, 200, 300],
   "colsample_bytree": [0.7, 1.0],
   "subsample": [0.7, 1.0]
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

    return best_models, gs_best.best_params_, y_pred

model, best_params, y_pred = hyperparameter_optimization(X_train, y_train, X_test, y_test)

# Metriklere ve gelişimlerine dayanarak XGBoost Classifier ana model olarak seçilmiştir.

# XGBoost modelini oluşturun
final_model = XGBClassifier(**best_params)

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

# Presicion-recall eğrisi:
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


# Feature Importance
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


# Top 15 önemli değişkenler
plot_importance(final_model, X, num=15)



# Burada X_testin segmentlerini kaydedip daha sonra öneri yapabilmek için churn edenlerin kaç tanesi hangi segmentte olduğunu yakalıyoruz
segment_info = df['Segment']
test_indices = X_test.index

test_segments = segment_info[test_indices]

results_df = pd.DataFrame({
    'Predicted Target': y_pred,
    'Actual Target': y_test,
    'Segment': test_segments})

correct_and_one = results_df[(results_df['Actual Target'] == 1) & (results_df['Predicted Target'] == 1)]

# Bu durumlara ait Segment değerlerinin sayısını hesaplama
segment_counts_one = correct_and_one['Segment'].value_counts()

print(segment_counts_one)
# Bu değerlerin burada yakaladık ve Öneriler sayfasında tekrar bu kadar kodu çalıştırmamak için manuel data oluşturduk orada.

