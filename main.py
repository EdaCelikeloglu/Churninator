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


warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("BankChurners.csv")

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
# Total_Relationship_Count: Total no. of products held by the customer. yani müşterinin aynı bankadan hem kredi kartı
#                           hem banka kartı ve farklı tipte hesapları olabilir savings account gibi
# Months_Inactive_12_mon: müşterinin son 12 ayda kaç ay inactive kaldığının sayısı
# Contacts_Count_12_mon: müşteriyle son 12 ayda kurulan iletişim sayısı
# Credit_Limit: müşterinin kredi kartının limiti
# Total_Revolving_Bal: devir bakiyesi (Bu terim, müşterinin ödeme yapması gereken ancak henüz ödenmemiş olan borç
# #                     miktarını ifade eder. Yani, müşterinin kredi kartı hesabında biriken ve henüz ödenmemiş olan borç tutarıdır.)
# Avg_Open_To_Buy:  müşterinin ulaşabileceği maksimum kredi miktarının son 12 aydaki ortalaması
# Total_Amt_Chng_Q4_Q1: Change in Transaction Amount (Q4 over Q1)
# Total_Trans_Amt: son 12 aydaki tüm transaction'lardan gelen miktar
# Total_Trans_Ct: son 12 aydaki toplam transaction sayısı
# Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1)
# Avg_Utilization_Ratio: müşterinin mevcut kredi kartı borçlarının kredi limitine oranını ifade eder

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
cat_cols, num_cols, cat_but_car = grab_col_names(df)
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


df.drop(["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
         "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], inplace=True, axis=1)



# Değişkenlerin özet grafikleri
for col in num_cols:
    num_summary(df, col, plot=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)



# Base model
df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()
df.shape

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(df_scaled, columns=df[num_cols].columns)

y = df["Attrition_Flag_Existing Customer"]
X = df.drop(["Attrition_Flag_Existing Customer", "CLIENTNUM"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# sm = SMOTE(random_state=69, sampling_strategy=1.0)
#
# X_train, y_train = sm.fit_resample(X_train, y_train)

def base_models(X, y, scoring="accuracy"):
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
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                    ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=10, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


base_models(X, y)

#########################################################################################################################
df = pd.read_csv("BankChurners.csv")

df.drop(["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
         "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], inplace=True, axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Bağımlı değişkenimizin ismini target yapalım
df.rename(columns={"Attrition_Flag":"Target"}, inplace=True)

# ID kolonunda duplicate bakıp, sonra bu değişkeni silme
df["CLIENTNUM"].nunique() # 10127 - yani duplicate yok id'de
df.drop("CLIENTNUM", axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# outliers
# IQR
for col in num_cols:
    print(col)
    grab_outliers(df, col)


############################# encode etmeden, IQR yapmadan, sırf num_cols'da LOF
# LOF - string ile çalışmıyor
# pca- temel bileşen analizi, 100 değişken varken 2 değişkene indirgem

# bakalım çok değişkenli yaklaştığımızda ne olacak
# buradaki komşuluk sayısı 20, default da 20 zaten
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols]) # skorları getirir

# skorları tutma
df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores -> eğer eksi değerleriyle değerlendirmek istemezsen, skorları pozitife çevirir
np.sort(df_scores)[0:5] # en kötü 5 gözlem


# elbow (dirsek) yöntemi
# her bir nokta eşik değerini temsil ediyor
# en marjinal değişiklik, kırılım, nerede olduysa onu eşik değer olarak belirleyebiliriz
# mesela burada 3. index'teki değeri seçmeyi tercih edebiliriz
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 100], style='.-')
plt.show()

th = np.sort(df_scores)[25]

# -4'ten daha küçük yani -5,-6 gibi değerleri seçme
df[df_scores < th]

df[df_scores < th].shape
# bunların neden aykırı olduğunu anlamak istersek:
# özet istatistikleriyle kıyaslayarak anlam çıkarabiliriz
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

#outlier
for col in num_cols:
    print(col)
    grab_outliers(df, col)

for col in num_cols:
    replace_with_thresholds(df, col)

# NaN işlemleri
cols_with_unknown = ['Income_Category', "Education_Level"]
for col in cols_with_unknown:
    df[col] = df[col].apply(lambda x: np.nan if x == 'Unknown' else x)

# Encoding işlemleri
df["Target"] = df.apply(lambda x: 0 if (x["Target"] == "Existing Customer") else 1, axis=1)

df.head()
df.shape # (10127, 21)

# Gender
df["Gender"] = df.apply(lambda x: 1 if (x["Gender"] == "F") else 0, axis=1)
df["Gender"].unique()


# ordinal encoder
def ordinal_encoder(dataframe, col):
    edu_cats = ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', np.nan]
    income_cats = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', np.nan]
    col_cats = []

    if col is "Education_Level":
        col_cats = edu_cats
    if col is "Income_Category":
        col_cats = income_cats

    # cat_codes = range(0, len(col_cats)) # buna ihtiyaç var mı? sanki kendisi rakamları veriyor gibi.
    ordinal_encoder = OrdinalEncoder(categories=[col_cats])
    df[col] = ordinal_encoder.fit_transform(df[[col]])

    print(df[col].head(20))
    return df

df = ordinal_encoder(df,"Education_Level")
df = ordinal_encoder(df,"Income_Category")

# one-hot encoder
df = one_hot_encoder(df, ["Marital_Status", "Card_Category"], drop_first=True)

# knn'in uygulanması. knn komşuların ortalamasıyla doldurur
dff = df.copy()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=10)
# dff["Income_Category"] = pd.DataFrame(imputer.fit_transform(dff["Income_Category"]), columns=dff.columns)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


dff["Education_Level"] = dff["Education_Level"].round().astype(int)
dff["Income_Category"] = dff["Income_Category"].round().astype(int)


# yeni değişkenler
df.head()

df.groupby("Contacts_Count_12_mon")["Months_Inactive_12_mon"].mean()
df.groupby("Months_Inactive_12_mon")["Contacts_Count_12_mon"].mean()

labels = ['Young', 'Middle Aged', 'Senior']
bins = [25, 35, 55, 74]
df['Customer_Age_Category'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels)

df.groupby("Card_Category")["Customer_Age_Category"].count()

# kart grubunda yaş kategorilerine bakma
count_by_card_age_category = df.groupby("Card_Category")["Customer_Age_Category"].value_counts()


total_counts_by_card = df.groupby("Card_Category")["Customer_Age_Category"].count()
percentage_by_card_age_category = count_by_card_age_category.div(total_counts_by_card, level='Card_Category') * 100
print("Count:")
print(count_by_card_age_category)
print("\nPercentage:")
print(percentage_by_card_age_category)

# Kart grubu kırılımında target
count_by_card_target_age_category = df.groupby(["Card_Category", "Target"])["Customer_Age_Category"].value_counts().unstack(fill_value=0)
total_counts_by_card_target = df.groupby(["Card_Category", "Target"])["Customer_Age_Category"].count().unstack(fill_value=0)
print("Count:")
print(count_by_card_target_age_category)


# Yüzdelikli bakış
# Hedef değişkenin yüzdelerini hesaplayalım
percentage_by_card_target_age_category = count_by_card_target_age_category.div(total_counts_by_card_target.sum(axis=1), axis=0) * 100
print("Percentage by Target:")
print(percentage_by_card_target_age_category)

# churn etme olasılıkları, kart kategrisi ve yaş grubu kırılımında
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
ratios = grouped_counts *100 / total_counts
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
income_cat_target_credit_limit = df.groupby(["Target","Income_Category" ])["Income_Category"].count()


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

df.head()


# total_revolving_bal / credit_limit = Avg_Utilization_Ratio?

df["uti_rate_out_calc"] =  df["Total_Revolving_Bal"] / df["Credit_Limit"]

df[["Avg_Utilization_Ratio",  "uti_rate_out_calc"]].head(20)




a = df["uti_rate_out_calc"] == df["Avg_Utilization_Ratio"]

a.head(20)



result = df["Avg_Utilization_Ratio"].equals(df["Total_Revolving_Bal"] / df["Credit_Limit"])
# bu false çıktı ama virgül sonrası sebebiyle  bu yüzden bu değişkenin anlamlı olmayacağına karar verdik

df.drop("uti_rate_out_calc", inplace=True, axis=1)

df.head()


# smote
# bunları çalıştırabilmek için base model için hazırladığımız df'i kullandım

# Applying SMOTE to handle imbalance in target variable
from imblearn.over_sampling import SMOTE
# SMOTE yöntemini uygulayalım
sm = SMOTE(random_state=69, sampling_strategy=1.0)

X.head()
y.head()

X.columns

# SMOTE yöntemini uygulayalım
sm = SMOTE(random_state=69, sampling_strategy=1.0)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Yeni örnekleme sonuçlarını kullanarak bir DataFrame oluşturalım (isteğe bağlı)
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df["Attrition_Flag_Existing Customer"] = y_resampled

# Sonuçları inceleyelim
print("Orjinal veri çerçevesi boyutu:", df.shape) # Orjinal veri çerçevesi boyutu: (10127, 52)
print("Yeniden örnekleme sonrası veri çerçevesi boyutu:", resampled_df.shape) # Yeniden örnekleme sonrası veri çerçevesi boyutu: (17000, 51)


resampled_df["Attrition_Flag_Existing Customer"].value_counts()
# Attrition_Flag_Existing Customer
# 1    8500
# 0    8500

# kitaptaki smote
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where

y = df["Target"]
X = df.drop(["Target", "CLIENTNUM"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

counter = Counter(y)
print(counter)

# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)



# kaggle'da biri şöyle yapmış
# Applying SMOTE to handle imbalance in target variable

sm = SMOTE(random_state=69, sampling_strategy=1.0)

X_train, y_train = sm.fit_resample(X_train, y_train)


# Cost sensitive learning
# Weighted Logistic Regression
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# define model
weights = {0: 0.01, 1: 1.0}
model = LogisticRegression(solver='lbfgs', class_weight=weights)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
# Mean ROC AUC: 0.911



# confusion matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))
