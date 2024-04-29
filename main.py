import graphviz
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
                             classification_report, RocCurveDisplay)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from xgboost import XGBClassifier
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


import numpy as np
import pandas as pd
import missingno as msno
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
#from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
#from missingpy import MissForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("BankChurners.csv")
df.head()

df.shape

df.columns

df.info()

df.columns
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
# Total_Amt_Chng_Q4_Q1: transaction sayısındaki 4. çeyrek ve 1. çeyrek arasındaki fark
# Total_Trans_Amt: son 12 aydaki tüm transaction'lardan gelen miktar
# Total_Trans_Ct: son 12 aydaki toplam transaction sayısı
# Total_Ct_Chng_Q4_Q1: transaction miktarlarının 4. çeyrek ve 1. çeyrek arasındaki fark
# Avg_Utilization_Ratio: müşterinin mevcut kredi kartı borçlarının kredi limitine oranını ifade eder

#fonksiyonlarımız
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

df.drop(["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
         "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], inplace=True, axis=1)

df.info()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols

num_cols

df.nunique()

for col in num_cols:
    num_summary(df, col, plot=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

df.head()

# Base model
df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(df_scaled, columns=df[num_cols].columns)

df.head()


y = df["Attrition_Flag_Existing Customer"]
X = df.drop(["Attrition_Flag_Existing Customer", "CLIENTNUM"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


base_models(X, y, scoring="accuracy")




X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=42)
log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

df.head()
#33333333333333333333333333333333333333333333333333333333333333333

df = pd.read_csv("BankChurners.csv")

df.drop(["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
         "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], inplace=True, axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df[cat_cols].head(20)

# Bağımlı değişkenimizin ismini target yapalım
#df["Target"] = df["Attrition_Flag"]
#df["Target"].unique()
#df.head()
# df.drop("Attrition_Flag", axis=1, inplace=True)
df.rename(columns={"Attrition_Flag":"Target"}, inplace=True)

df["CLIENTNUM"].nunique() # 10127 - yani duplicate yok id'de
df.drop("CLIENTNUM", axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# NaN işlemleri
cols_with_unknown = ['Income_Category', "Marital_Status", "Education_Level"]
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
    edu_cats = ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', 'Unknown']
    income_cats = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown']
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

"""
df["Education_Level"].unique()
cats = ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', 'Unknown']
category_codes = [0, 1, 2, 3, 4, 5, 6]
ordinal_encoder = OrdinalEncoder(categories=[cats])
df["Education_Level"] = ordinal_encoder.fit_transform(df[['Education_Level']])


from sklearn.preprocessing import OrdinalEncoder
income_cats = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown']
income_codes = [0, 1, 2, 3, 4, 5]

ordinal_encoder = OrdinalEncoder(categories=[income_cats])
df["Income_Category"] = ordinal_encoder.fit_transform(df[['Income_Category']])

df["Income_Category"].head(20)

df["Income_Category"].value_counts()
df["Income_Category"].isnull().sum() # 1112
# 0.000    3561
# 1.000    1790
# 3.000    1535
# 2.000    1402
# 4.000     727
"""

# knn'in uygulanması. knn komşuların ortalamasıyla doldurur
dff = df.copy()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=1)
dff["Income_Category"] = pd.DataFrame(imputer.fit_transform(dff["Income_Category"]), columns=dff.columns)
#dff["Income_Category"] = pd.DataFrame(imputer.fit_transform(dff["Income_Category"]))
dff.head()


dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()











df["Income_Category"].unique()

df.head()

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





# ağaç yöntemlerinde çok dokunulmanması öneriliyor, en kötü ucundan dokunulması gerek





############################# encode ettikten sonra LOF
