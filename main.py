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

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

base = pd.read_csv("BankChurners.csv")

# CLIENTNUM: müşteri id'si
# Attrition_Flag: TARGET. Churn etti mi etmedi mi bilgisine sahip. (kaggle'da şöyle yazmışlar: if the account is closed then 1 else 0)
# Customer_Age: müşterinin yaşı
# Gender: müşterinin cinsiyeti (F, M)
# Dependent_count: müşterinin bakmakla yükümlü olduğu kişi sayısı
# Education_Level: eğitim seviyesi (High School, Graduate, Uneducated, Unknown, College, Post-Graduate, Doctorate)
# Marital_Status: müşterinin medeni durumu (Married, Single, Unknown, Divorced)
# Income_Category: müşterinin hangi gelir kategorisinde olduğu bilgisi ($60K - $80K, Less than $40K, $80K - $120K, $40K - $60K, $120K +, Unknown)
# Card_Category: müşterinin sahip olduğu kartın türü (Blue, Silver, Gold, Platinum)
# Months_on_book: müşteri kaç aydır bu bankada
# * Total_Relationship_Count: Total no. of products held by the customer. yani müşterinin aynı bankadan hem kredi kartı
#                           hem banka kartı ve farklı tipte hesapları olabilir savings account gibi
# * Months_Inactive_12_mon: müşterinin son 12 ayda kaç ay inactive kaldığının sayısı
# Contacts_Count_12_mon: müşteriyle son 12 ayda kurulan iletişim sayısı
# Credit_Limit: müşterinin kredi kartının limiti
# * Total_Revolving_Bal: devir bakiyesi (Bu terim, müşterinin ödeme yapması gereken ancak henüz ödenmemiş olan borç
# #                    miktarını ifade eder. Yani, müşterinin kredi kartı hesabında biriken ve henüz ödenmemiş olan borç tutarıdır.)
# Avg_Open_To_Buy:  müşterinin ulaşabileceği maksimum kredi miktarının son 12 aydaki ortalaması
# Total_Amt_Chng_Q4_Q1: Change in Transaction Amount (Q4 over Q1)
# Total_Trans_Amt: son 12 aydaki tüm transaction'lardan gelen miktar
# * Total_Trans_Ct: son 12 aydaki toplam transaction sayısı
# Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1)
# * Avg_Utilization_Ratio: müşterinin mevcut kredi kartı borçlarının kredi limitine oranını ifade eder

# Fonksiyonlarımız
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
cat_cols, num_cols, cat_but_car = grab_col_names(base)

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

base = pd.read_csv("BankChurners.csv")
base.head()
base.drop([
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"],
    inplace=True, axis=1)

# Bağımlı değişkenimizin ismini target yapalım
base.rename(columns={"Attrition_Flag": "Target"}, inplace=True)
base["Target"] = base.apply(lambda x: 0 if (x["Target"] == "Existing Customer") else 1, axis=1)

# ID kolonunda duplicate bakıp, sonra bu değişkeni silme
base["CLIENTNUM"].nunique()  # 10127 - yani duplicate yok id'de
base.drop("CLIENTNUM", axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(base)
cat_cols = [col for col in cat_cols if col not in "Target"]

# Değişkenlerin özet grafikleri
for col in num_cols:
    num_summary(base, col, plot=True)

for col in cat_cols:
    cat_summary(base, col, plot=True)

# Base model
base = one_hot_encoder(base, cat_cols, drop_first=True)
base.head()
base.shape


scaler = StandardScaler()
base_scaled = scaler.fit_transform(base[num_cols])
base[num_cols] = pd.DataFrame(base_scaled, columns=base[num_cols].columns)

y_base = base["Target"]
X_base = base.drop("Target", axis=1)

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

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

model_metrics(X_train_base, y_train_base, X_test_base, y_test_base)

#smotesuz hali
'''
Classification Report for LR:
              precision    recall  f1-score   support
           0       0.92      0.97      0.95      1699
           1       0.77      0.59      0.67       327
    accuracy                           0.91      2026
   macro avg       0.85      0.78      0.81      2026
weighted avg       0.90      0.91      0.90      2026
Classification Report for KNN:
              precision    recall  f1-score   support
           0       0.91      0.98      0.94      1699
           1       0.81      0.52      0.63       327
    accuracy                           0.90      2026
   macro avg       0.86      0.75      0.79      2026
weighted avg       0.90      0.90      0.89      2026
Classification Report for SVC:
              precision    recall  f1-score   support
           0       0.94      0.98      0.96      1699
           1       0.87      0.70      0.77       327
    accuracy                           0.93      2026
   macro avg       0.91      0.84      0.87      2026
weighted avg       0.93      0.93      0.93      2026
Classification Report for CART:
              precision    recall  f1-score   support
           0       0.96      0.96      0.96      1699
           1       0.79      0.78      0.78       327
    accuracy                           0.93      2026
   macro avg       0.87      0.87      0.87      2026
weighted avg       0.93      0.93      0.93      2026
Classification Report for RF:
              precision    recall  f1-score   support
           0       0.95      0.99      0.97      1699
           1       0.93      0.72      0.81       327
    accuracy                           0.95      2026
   macro avg       0.94      0.85      0.89      2026
weighted avg       0.95      0.95      0.94      2026
Classification Report for Adaboost:
              precision    recall  f1-score   support
           0       0.96      0.98      0.97      1699
           1       0.88      0.79      0.83       327
    accuracy                           0.95      2026
   macro avg       0.92      0.88      0.90      2026
weighted avg       0.95      0.95      0.95      2026
Classification Report for GBM:
              precision    recall  f1-score   support
           0       0.97      0.99      0.98      1699
           1       0.93      0.83      0.88       327
    accuracy                           0.96      2026
   macro avg       0.95      0.91      0.93      2026
weighted avg       0.96      0.96      0.96      2026
Classification Report for XGBoost:
              precision    recall  f1-score   support
           0       0.97      0.98      0.98      1699
           1       0.89      0.86      0.87       327
    accuracy                           0.96      2026
   macro avg       0.93      0.92      0.92      2026
weighted avg       0.96      0.96      0.96      2026

Classification Report for LightGBM:
              precision    recall  f1-score   support
           0       0.97      0.98      0.98      1699
           1       0.91      0.87      0.89       327
    accuracy                           0.96      2026
   macro avg       0.94      0.92      0.93      2026
weighted avg       0.96      0.96      0.96      2026
Classification Report for CatBoost:
              precision    recall  f1-score   support
           0       0.98      0.98      0.98      1699
           1       0.92      0.88      0.90       327
    accuracy                           0.97      2026
   macro avg       0.95      0.93      0.94      2026
weighted avg       0.97      0.97      0.97      2026


'''



counter = Counter(y_train_base)
print(counter)

Counter(y_test_base)

# transform the dataset
oversample = SMOTE()
X_train_base, y_train_base = oversample.fit_resample(X_train_base, y_train_base)
# summarize the new class distribution
counter = Counter(y_train_base)
print(counter)


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


model_metrics(X_train_base, y_train_base, X_test_base, y_test_base)
"""
Base Models....
Classification Report for LR:
              precision    recall  f1-score   support

           0       0.94      0.95      0.94      1699
           1       0.70      0.66      0.68       327

    accuracy                           0.90      2026
   macro avg       0.82      0.80      0.81      2026
weighted avg       0.90      0.90      0.90      2026

Classification Report for KNN:
              precision    recall  f1-score   support

           0       0.98      0.81      0.89      1699
           1       0.48      0.89      0.62       327

    accuracy                           0.83      2026
   macro avg       0.73      0.85      0.76      2026
weighted avg       0.90      0.83      0.84      2026

Classification Report for SVC:
              precision    recall  f1-score   support

           0       0.95      0.97      0.96      1699
           1       0.83      0.75      0.79       327

    accuracy                           0.93      2026
   macro avg       0.89      0.86      0.87      2026
weighted avg       0.93      0.93      0.93      2026

Classification Report for CART:
              precision    recall  f1-score   support

           0       0.97      0.95      0.96      1699
           1       0.75      0.84      0.80       327

    accuracy                           0.93      2026
   macro avg       0.86      0.90      0.88      2026
weighted avg       0.93      0.93      0.93      2026

Classification Report for RF:
              precision    recall  f1-score   support

           0       0.97      0.98      0.97      1699
           1       0.87      0.85      0.86       327

    accuracy                           0.96      2026
   macro avg       0.92      0.91      0.92      2026
weighted avg       0.96      0.96      0.96      2026

Classification Report for Adaboost:
              precision    recall  f1-score   support

           0       0.98      0.96      0.97      1699
           1       0.80      0.88      0.84       327

    accuracy                           0.95      2026
   macro avg       0.89      0.92      0.90      2026
weighted avg       0.95      0.95      0.95      2026

Classification Report for GBM:
              precision    recall  f1-score   support

           0       0.98      0.96      0.97      1699
           1       0.83      0.91      0.87       327

    accuracy                           0.96      2026
   macro avg       0.91      0.94      0.92      2026
weighted avg       0.96      0.96      0.96      2026

Classification Report for XGBoost:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98      1699
           1       0.88      0.90      0.89       327

    accuracy                           0.96      2026
   macro avg       0.93      0.94      0.94      2026
weighted avg       0.97      0.96      0.97      2026

Classification Report for LightGBM:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98      1699
           1       0.89      0.91      0.90       327

    accuracy                           0.97      2026
   macro avg       0.93      0.94      0.94      2026
weighted avg       0.97      0.97      0.97      2026

Classification Report for CatBoost:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98      1699
           1       0.89      0.90      0.90       327

    accuracy                           0.97      2026
   macro avg       0.94      0.94      0.94      2026
weighted avg       0.97      0.97      0.97      2026
"""

#########################################################################################################################
df = pd.read_csv("BankChurners.csv")

df.drop([
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"],
    inplace=True, axis=1)

# Bağımlı değişkenimizin ismini target yapalım
df.rename(columns={"Attrition_Flag": "Target"}, inplace=True)

# ID kolonunda duplicate bakıp, sonra bu değişkeni silme
df["CLIENTNUM"].nunique()  # 10127 - yani duplicate yok id'de
df.drop("CLIENTNUM", axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Outliers
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

# NaN işlemleri
cols_with_unknown = ['Income_Category', "Education_Level"]
for col in cols_with_unknown:
    df[col] = df[col].apply(lambda x: np.nan if x == 'Unknown' else x)


# Encoding işlemleri
df["Target"] = df.apply(lambda x: 0 if (x["Target"] == "Existing Customer") else 1, axis=1)


# Ordinal encoder
def ordinal_encoder(dataframe, col):
    edu_cats = ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', np.nan]
    income_cats = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', np.nan]
    customer_age_cat = [ 'Young','Middle_Aged', 'Senior']

    if col is "Education_Level":
        col_cats = edu_cats
    if col is "Income_Category":
        col_cats = income_cats
    if col is "Customer_Age_Category":
        col_cats = customer_age_cat

    ordinal_encoder = OrdinalEncoder(categories=[col_cats])  # burada direkt int alamıyorum çünkü NaN'lar mevcut.
    df[col] = ordinal_encoder.fit_transform(df[[col]])

    print(df[col].head(20))
    return df


df = ordinal_encoder(df, "Education_Level")
df = ordinal_encoder(df, "Income_Category")
df = ordinal_encoder(df, "Customer_Age_Category")


# Yeni değişkenler
df.groupby("Contacts_Count_12_mon")["Months_Inactive_12_mon"].mean()
df.groupby("Months_Inactive_12_mon")["Contacts_Count_12_mon"].mean()

labels = ['Young', 'Middle_Aged', 'Senior']
bins = [25, 35, 55, 74]
df['Customer_Age_Category'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels)


###
def combine_categories(df, cat_col1, cat_col2, new_col_name):
    df[new_col_name] = df[cat_col1].astype(str) + '_' + df[cat_col2].astype(str)


combine_categories(df, 'Customer_Age_Category', 'Marital_Status', 'Age_&_Marital')
combine_categories(df, 'Gender', 'Customer_Age_Category', 'Gender_&_Age')
combine_categories(df, "Card_Category", "Customer_Age_Category", "Card_&_Age")
combine_categories(df, "Gender", "FrequencyScore", "Gender_&_Frequency")
combine_categories(df, "Gender", "MonetaryScore", "Gender_&_Monetary")


df["Gender_&_Frequency"].value_counts()

df.groupby("Card_&_Age")["Target"].mean()


# kart grubunda yaş kategorilerine bakma
count_by_card_age_category = df.groupby("Card_Category")["Customer_Age_Category"].value_counts()

total_counts_by_card = df.groupby("Card_Category")["Customer_Age_Category"].count()
percentage_by_card_age_category = count_by_card_age_category.div(total_counts_by_card, level='Card_Category') * 100
print("Count:")
print(count_by_card_age_category)
print("\nPercentage:")
print(percentage_by_card_age_category)

# Kart grubu kırılımında target
count_by_card_target_age_category = df.groupby(["Card_Category", "Target"])[
    "Customer_Age_Category"].value_counts().unstack(fill_value=0)
total_counts_by_card_target = df.groupby(["Card_Category", "Target"])["Customer_Age_Category"].count().unstack(
    fill_value=0)
print("Count:")
print(count_by_card_target_age_category)

# Yüzdelikli bakış
# Hedef değişkenin yüzdelerini hesaplayalım
percentage_by_card_target_age_category = count_by_card_target_age_category.div(total_counts_by_card_target.sum(axis=1),
                                                                               axis=0) * 100
print("Percentage by Target:")
print(percentage_by_card_target_age_category)

# Churn etme olasılıkları, kart kategrisi ve yaş grubu kırılımında
count_by_credit_limit = df.groupby(["Card_Category", "Customer_Age_Category"])["Target"].mean()
# Blue           Young                   0.132
#                Middle Aged             0.165
#                Senior                  0.159
# Gold           Young                   0.300
#                Middle Aged             0.170
#                Senior                  0.167
# Platinum       Young                     NaN
#                Middle Aged             0.263
#                Senior                  0.000
# Silver         Young                   0.118
#                Middle Aged             0.152
#                Senior                  0.140


count_by_credit_limit = df.groupby(["Card_Category"])["Target"].mean()
# TODO Plat olanların %25'i churn etmiş

count_by_credit_limit = df.groupby(["Target"])["Card_Category"].value_counts()
# TODO (oversampling sonrası değişebilir) gold ve plat rare encoder ile birleştirilebilir.

# Kart grubu kırılımında limit
count_by_credit_limit = df.groupby("Card_Category")["Credit_Limit"].mean()

# Kart grubu kırılımında limit ve target
count_by_credit_limit = df.groupby(["Card_Category", "Target"])["Credit_Limit"].mean()

# Medeni durumun kırılımında bakmakla yükümlü olunan insan sayısı analizi
marital_status_dependents = df.groupby("Marital_Status")["Dependent_count"].mean()

# Dependent kırılımında kredi limiti analizi
df["Dependent_count"].value_counts()
dependent_count_credit_limit = df.groupby("Dependent_count")["Credit_Limit"].mean()
# TODO dependent sayısı arttıkça limit artıyor

# Müşterinin kaldığı ay sayısı ve kart kategorisi
months_on_book_card_category = df.groupby("Card_Category")["Months_on_book"].mean()
# Anlamlı bir sonuç çıkmadı


# Eğitim ve kart kategorisi analizi
df["Education_Level"].value_counts()
education_card_category = df.groupby("Education_Level")["Card_Category"].value_counts()

# Müşteriyle iletişime geçme ve target
df["Contacts_Count_12_mon"].value_counts()

contacts_target = df.groupby("Target")["Contacts_Count_12_mon"].value_counts()
df.groupby("Target")["Contacts_Count_12_mon"].count() / (len(df["Target"]))

# Target'e göre Contacts_Count_12_mon oranları
total_counts = df.groupby("Target")["Contacts_Count_12_mon"].count()
grouped_counts = df.groupby(["Target", "Contacts_Count_12_mon"]).size()
ratios = grouped_counts * 100 / total_counts
print(ratios)
"""
1       0                        0.430
        1                        6.638
        2                       24.770
        3                       41.856 # TODO
        4                       19.361
        5                        3.626
        6                        3.319
"""

# Hedef değişkenine göre Contacts_Count_12_mon değerlerinin yüzdelerini hesaplayalım
percentage_by_target_contacts = df.groupby("Target")["Contacts_Count_12_mon"].value_counts(normalize=True).mul(100)

# Sonucu ekrana yazdıralım
print("Percentage by Target and Contacts_Count_12_mon:")
print(percentage_by_target_contacts)

# Utilization ratio ve target analizi

# Hedef değişkenine göre Avg_Utilization_Ratio sütununun ortalama değerlerini hesaplayalım
mean_by_target_ratio = df.groupby("Target")["Avg_Utilization_Ratio"].mean()
# TODO churn edenlerin borç ortalaması daha azmış
# borç ödemeleri daha kolay -- borç/limit oranı daha küçük. daha ödenebilir krediler çekilmiş.
# çektiği kredi, maaşına(limite) oranla daha fazla olanlar, bankaya ödeme yapmaya devam ettikleri için bankayı terk edemiyor olabilirler mi??


df.head()

# Gelir grubu ve target kırılımında limit analizi
income_cat_target_credit_limit = df.groupby(["Target", "Income_Category"])["Income_Category"].count()

# Gelir kategorileri 0 olan hedef değişkeni için sayıları hesaplayalım
count_by_income_target = df.groupby(["Income_Category", "Target"])["Income_Category"].count().unstack(fill_value=0)

# 0 hedef değişkeni için gelir kategorilerinin sayılarını alalım
count_0_target_income = count_by_income_target[0]

# 0 hedef değişkeni için gelir kategorilerinin yüzdelerini hesaplayalım
percentage_0_target_income = count_0_target_income / count_0_target_income.sum() * 100

# Sonucu birleştirelim
result = pd.DataFrame({"Count_Target_0": count_0_target_income, "Percentage_Target_0": percentage_0_target_income})

# Sonucu ekrana yazdıralım
print(result)

# Gelir kategorileri 1 olan hedef değişkeni için sayıları hesaplayalım
# Gelir kategorileri için hem 0 hem de 1 hedef değişkeni için sayıları hesaplayalım
count_by_income_target = df.groupby(["Income_Category", "Target"])["Income_Category"].count().unstack(fill_value=0)

# 1 hedef değişkeni için gelir kategorilerinin sayılarını alalım
count_1_target_income = count_by_income_target[1]

# 1 hedef değişkeni için gelir kategorilerinin yüzdelerini hesaplayalım
percentage_1_target_income = count_1_target_income / count_1_target_income.sum() * 100

# Sonucu birleştirelim
result = pd.DataFrame({"Count_Target_1": count_1_target_income, "Percentage_Target_1": percentage_1_target_income})

# Sonucu ekrana yazdıralım
print(result)

# Cinsiyet ve target analizi
gender_target = df.groupby("Gender")["Target"].mean()

# FM analizi (FM skorları hesaplama)
# Total_Trans_Amt: son 12 aydaki tüm transaction'lardan gelen miktar
# Total_Trans_Ct: son 12 aydaki toplam transaction sayısı

# Frequency
df["MonetaryScore"] = pd.qcut(df["Total_Trans_Amt"], 5, labels=[1, 2, 3, 4, 5])
df["FrequencyScore"] = pd.qcut(df["Total_Trans_Ct"], 5, labels=[1, 2, 3, 4, 5])

df["FrequencyScore"].value_counts()
df["MonetaryScore"].value_counts()
df.groupby("FrequencyScore")["Target"].mean()
df.groupby("MonetaryScore")["Income_Category"].mean()
df.groupby("Income_Category")["Avg_Utilization_Ratio"].mean()
df.groupby(["Income_Category", "MonetaryScore"])["Target"].mean()

"""
MonetaryScore
1   0.204
2   0.439
3   0.008
4   0.022
5   0.130
"""

# Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1)
# Total_Amt_Chng_Q4_Q1: Change in Transaction Amount (Q4 over Q1)

# Çeyreklik farklılıklar arasındaki farkların analizi. oran olduğu için 1'i threshold olarak belirleyip yeni bir kategorik
# Değişken oluşturacağız

df["Total_Ct_Chng_Q4_Q1"].describe().T
df["Total_Amt_Chng_Q4_Q1"].describe().T

df['Total_Amt_Increased'] = np.where((df['Total_Amt_Chng_Q4_Q1'] > 0) & (df['Total_Amt_Chng_Q4_Q1'] < 1), 0, 1)
# 0: Q1'in fazla oldukları
# 1: Q4'ün fazla oldukları

df['Total_Ct_Increased'] = np.where((df['Total_Ct_Chng_Q4_Q1'] > 0) & (df['Total_Ct_Chng_Q4_Q1'] < 1), 0, 1)

df['Total_Amt_Increased'].value_counts()
df['Total_Ct_Increased'].value_counts()

# bu yeni değişkenlerin target ile analizi
ct_target = df.groupby("Total_Ct_Increased")["Target"].mean()
amt_target = df.groupby("Total_Amt_Increased")["Target"].mean()


# kategori incelemesi
for col in cat_cols:
    counts = df[col].value_counts()
    counts_under_30 = counts[counts < 30]
    if not counts_under_30.empty:
        print(counts_under_30)

# Card_Category
# Platinum    20

# Card_&_Age
# Platinum_Middle_Aged    19
# Gold_Young              10
# Gold_Senior              6
# Platinum_Senior          1

# Months_Inactive_12_mon
# 0    29

# platium 20 olduğu içim gold ile birleştiriyoruz:
df["Card_Category"] = df["Card_Category"].apply(lambda x: "Gold_Platinum" if x == "Platinum" or x == "Gold" else x)

# Months_Inactive_12_mon bunu da 29 olduğu için 0 olanlara 1 yazalım:
df["Months_Inactive_12_mon"] = df["Months_Inactive_12_mon"].apply(lambda x: 1 if x == 0 else x)


# Card & Age de Rare encoding yapalım:
df["Card_&_Age"] = df["Card_&_Age"].apply(lambda x: "Rare" if df["Card_&_Age"].value_counts()[x] < 30 else x)

df.head()


# one-hot encoder
df = one_hot_encoder(df, ["Gender", "Marital_Status", "Card_Category",
                          "Age_&_Marital", "Gender_&_Age", "Card_&_Age", "Gender_&_Frequency", "Gender_&_Monetary"], drop_first=True)




# Knn imputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=10)
# dff["Income_Category"] = pd.DataFrame(imputer.fit_transform(dff["Income_Category"]), columns=dff.columns)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


df.head(20)

df["Education_Level"] = df["Education_Level"].round().astype(int)
df["Income_Category"] = df["Income_Category"].round().astype(int)

# TODO üstteki ile birleştir.
for col in df.columns:
    if df[col].dtype == 'float64':  # Sadece float sütunları kontrol edelim
        if (df[col] % 1 == 000).all():  # Tüm değerlerin virgülden sonrası 0 mı kontrol edelim
            df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Robust scaler
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

# Smote
# Applying SMOTE to handle imbalance in target variable
dff = df.copy()
# kitaptaki smote

y = df["Target"]
X = df.drop(["Target"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

counter = Counter(y_train)
print(counter)

Counter(y_test)

# transform the dataset
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
# summarize the new class distribution
counter = Counter(y_train)
print(counter)


#bence burada tüm modellerimizi bir çalıştırmalıyız


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
"""
Classification Report for LR:
              precision    recall  f1-score   support

           0       0.95      0.88      0.91      1699
           1       0.54      0.74      0.62       327

    accuracy                           0.86      2026
   macro avg       0.74      0.81      0.77      2026
weighted avg       0.88      0.86      0.86      2026

Classification Report for KNN:
              precision    recall  f1-score   support

           0       0.97      0.88      0.92      1699
           1       0.58      0.87      0.70       327

    accuracy                           0.88      2026
   macro avg       0.78      0.88      0.81      2026
weighted avg       0.91      0.88      0.89      2026

Classification Report for SVC:
              precision    recall  f1-score   support

           0       0.97      0.92      0.94      1699
           1       0.66      0.83      0.74       327

    accuracy                           0.91      2026
   macro avg       0.82      0.88      0.84      2026
weighted avg       0.92      0.91      0.91      2026

Classification Report for CART:
              precision    recall  f1-score   support

           0       0.96      0.94      0.95      1699
           1       0.73      0.81      0.77       327

    accuracy                           0.92      2026
   macro avg       0.85      0.88      0.86      2026
weighted avg       0.93      0.92      0.92      2026

Classification Report for RF:
              precision    recall  f1-score   support

           0       0.98      0.97      0.97      1699
           1       0.85      0.89      0.87       327

    accuracy                           0.96      2026
   macro avg       0.91      0.93      0.92      2026
weighted avg       0.96      0.96      0.96      2026

Classification Report for Adaboost:
              precision    recall  f1-score   support

           0       0.98      0.94      0.96      1699
           1       0.76      0.91      0.83       327

    accuracy                           0.94      2026
   macro avg       0.87      0.93      0.90      2026
weighted avg       0.95      0.94      0.94      2026

Classification Report for GBM:
              precision    recall  f1-score   support

           0       0.99      0.96      0.97      1699
           1       0.81      0.94      0.87       327

    accuracy                           0.95      2026
   macro avg       0.90      0.95      0.92      2026
weighted avg       0.96      0.95      0.96      2026

Classification Report for XGBoost:
              precision    recall  f1-score   support

           0       0.98      0.97      0.98      1699
           1       0.86      0.91      0.89       327

    accuracy                           0.96      2026
   macro avg       0.92      0.94      0.93      2026
weighted avg       0.96      0.96      0.96      2026

Classification Report for LightGBM:
              precision    recall  f1-score   support

           0       0.98      0.97      0.98      1699
           1       0.85      0.91      0.88       327

    accuracy                           0.96      2026
   macro avg       0.92      0.94      0.93      2026
weighted avg       0.96      0.96      0.96      2026

Classification Report for CatBoost:
              precision    recall  f1-score   support

           0       0.98      0.97      0.98      1699
           1       0.87      0.92      0.89       327

    accuracy                           0.96      2026
   macro avg       0.93      0.95      0.94      2026
weighted avg       0.97      0.96      0.96      2026

""" # Sonuçlar

######################################################
# 4. Automated Hyperparameter Optimization
######################################################
"""Random Forest:

max_depth: [5, 10, 15, None]
max_features: [sqrt(n_features), log2(n_features), 0.5]
min_samples_split: [2, 5, 10]
n_estimators: [50, 100, 200, 300]
XGBoost:

learning_rate: [0.01, 0.05, 0.1, 0.5]
max_depth: [3, 5, 7, 10]
n_estimators: [50, 100, 200, 300]
LightGBM:

learning_rate: [0.01, 0.05, 0.1, 0.5]
n_estimators: [50, 100, 200, 300]
max_depth: [3, 5, 7, 10]"""

df.shape

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

classifiers = ([("RF", RandomForestClassifier(), rf_params),
                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
                ('LightGBM', LGBMClassifier(force_col_wise=True), lightgbm_params),
                ('GBM', GradientBoostingClassifier(), gbm_params),
                ('CatBoost', CatBoostClassifier(verbose=False)), catboost_params])


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

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


best_models = hyperparameter_optimization(X, y)

[col for col in df.columns if col not in 'Target']