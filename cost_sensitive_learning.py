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

df = pd.read_csv("BankChurners.csv")

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

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

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

# Gender
df["Gender"] = df.apply(lambda x: 1 if (x["Gender"] == "F") else 0, axis=1)


# Ordinal encoder
def ordinal_encoder(dataframe, col):
    edu_cats = ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', np.nan]
    income_cats = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', np.nan]

    if col is "Education_Level":
        col_cats = edu_cats
    if col is "Income_Category":
        col_cats = income_cats

    ordinal_encoder = OrdinalEncoder(categories=[col_cats])  # burada direkt int alamıyorum çünkü NaN'lar mevcut.
    df[col] = ordinal_encoder.fit_transform(df[[col]])

    print(df[col].head(20))
    return df


df = ordinal_encoder(df, "Education_Level")
df = ordinal_encoder(df, "Income_Category")

labels = ['Young', 'Middle Aged', 'Senior']
bins = [25, 35, 55, 74]
df['Customer_Age_Category'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels)

df["MonetaryScore"] = pd.qcut(df["Total_Trans_Amt"], 5, labels=[1, 2, 3, 4, 5])
df["FrequencyScore"] = pd.qcut(df["Total_Trans_Ct"], 5, labels=[1, 2, 3, 4, 5])


df['Total_Amt_Increased'] = np.where((df['Total_Amt_Chng_Q4_Q1'] > 0) & (df['Total_Amt_Chng_Q4_Q1'] < 1), 0, 1)
# 0: Q1'in fazla oldukları
# 1: Q4'ün fazla oldukları

df['Total_Ct_Increased'] = np.where((df['Total_Ct_Chng_Q4_Q1'] > 0) & (df['Total_Ct_Chng_Q4_Q1'] < 1), 0, 1)

df = one_hot_encoder(df, ["Marital_Status", "Card_Category", "Customer_Age_Category"], drop_first=True)

# Knn imputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=10)
# dff["Income_Category"] = pd.DataFrame(imputer.fit_transform(dff["Income_Category"]), columns=dff.columns)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


df["Education_Level"] = df["Education_Level"].round().astype(int)
df["Income_Category"] = df["Income_Category"].round().astype(int)

for col in df.columns:
    if df[col].dtype == 'float64':  # Sadece float sütunları kontrol edelim
        if (df[col] % 1 == 000).all():  # Tüm değerlerin virgülden sonrası 0 mı kontrol edelim
            df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Robust scaler
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

df.head()
df.isnull().sum()

# Undersampling

# Methods that Select Examples to Keep: iki sınıfı da eşit sayıya indirgiyor
#                                       hangi örneklerin kalması gerektiğine karar veriyor
# Methods that Select Examples to Delete: silinmesi gerekenleri  belirler

# Methods that Select Examples to Keep

# Near miss undersampling
# Near Miss Undersampling, az örneklenmiş sınıfın örneklerini çoğunluk sınıfına olan uzaklığına göre azaltmayı amaçlayan
# bir yöntemdir. Bu yöntem, az sayıda olan sınıfın örneklerini, çoğunluk sınıfına olan mesafelerine göre "yakın" olanları
# koruyarak azaltır. Near Miss Undersampling'in üç farklı türü vardır:

# NearMiss-1: Bu yöntemde, çoğunluk sınıfına ait örnekler, azınlık sınıfına ait üç en yakın örnek arasındaki ortalama
# mesafeye göre seçilir. Yani, azınlık sınıfına ait örneklerle çoğunluk sınıfı arasındaki ortalama mesafe en küçük olan
# çoğunluk sınıfına ait örnekler seçilir.
# NearMiss-2: Bu yöntemde, çoğunluk sınıfına ait örnekler, azınlık sınıfına ait üç en uzak örnek arasındaki ortalama
# mesafeye göre seçilir. Yani, azınlık sınıfına ait örneklerle çoğunluk sınıfı arasındaki ortalama mesafe en küçük olan
# çoğunluk sınıfına ait örnekler seçilir.
# NearMiss-3: Bu yöntemde, çoğunluk sınıfına ait örnekler, her bir azınlık sınıfı örneğine olan en küçük mesafeye göre
# seçilir. Yani, her bir azınlık sınıfı örneğiyle en yakın mesafedeki çoğunluk sınıfı örneği seçilir.

# Yani, temelde Near Miss Undersampling, az sayıda olan sınıfın örneklerini seçerken, çoğunluk sınıfına olan uzaklıklarını
# dikkate alır ve bu uzaklık ölçüsüne göre örnekleri seçer. Bu sayede, sınıf dengesizliğini azaltmayı ve daha dengeli bir
# veri kümesi elde etmeyi amaçlar.

y = df["Target"]
X = df.drop("Target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Undersample imbalanced dataset with NearMiss-1
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot
from numpy import where
# summarize class distribution
counter = Counter(y)
print(counter) # {0: 8500, 1: 1627}
# define the undersampling method
undersample = NearMiss(version=1, n_neighbors=3)
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter) # {0: 1627, 1: 1627}


y = df["Target"]
X = df.drop("Target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Undersample imbalanced dataset with NearMiss-2
# summarize class distribution
counter = Counter(y)
print(counter) # {0: 8500, 1: 1627}
# define the undersampling method
undersample = NearMiss(version=2, n_neighbors=3)
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)
# {0: 1627, 1: 1627}


y = df["Target"]
X = df.drop("Target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Undersample imbalanced dataset with NearMiss-3
# summarize class distribution
counter = Counter(y)
print(counter) # {0: 8500, 1: 1627}
# define the undersampling method
undersample = NearMiss(version=3, n_neighbors_ver3=3)
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter) # {0: 1627, 1: 1627}


# Condensed Nearest Neighbor Rule Undersampling

# İşleyişi şöyle özetlenebilir:
# Başlangıçta, veri kümesindeki tüm örneklerden oluşan bir alt küme seçilir.
# Seçilen bu alt küme içindeki her bir örnek için, bu örneğe en yakın komşusu belirlenir.
# Eğer bir örnek ve en yakın komşusu aynı sınıfa ait değilse, o örnek alt kümeden çıkarılır.
# Tüm örnekler için bu işlem tekrarlanır ve alt küme, veri kümesinin çoğunluk sınıfını iyi temsil eden, ancak az sayıda
# olan sınıfı temsil eden örnekleri koruyacak şekilde güncellenir.

y = df["Target"]
X = df.drop("Target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the undersampling method
from imblearn.under_sampling import CondensedNearestNeighbour
undersample = CondensedNearestNeighbour(n_neighbors=1)
# summarize class distribution
counter = Counter(y)
print(counter)
# {0: 8500, 1: 1627}
# define the undersampling method
undersample = CondensedNearestNeighbour(n_neighbors=1)
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter) # {1: 1627, 0: 1450}

# Methods that Select Examples to Delete

# Tomek Links for Undersampling

# Tomek Links for Undersampling (Tomek Bağlantıları Azaltma) yöntemi, az sayıda olan sınıfın örneklerini korurken,
# çoğunluk sınıfından gereksiz örnekleri çıkarmayı amaçlayan bir tekniktir. Bu yöntem, sınıflar arasındaki sınırı
# netleştirmeye yöneliktir.
# İşleyişi şu şekildedir:
# Başlangıçta, veri kümesindeki her bir örneğin en yakın komşusu aranır.
# Eğer bir örnek ve en yakın komşusu birbirlerine ait değilse, yani farklı sınıflara aitlerse, bu iki örnek arasında
# bir "Tomek bağlantısı" oluşur. Yani, sınıflar arasındaki sınırda bulunan ve birbirine en yakın olan örnekler bir
# Tomek bağlantısı oluşturur.
# Bu Tomek bağlantıları, az sayıda olan sınıfın örneklerini korumak için kullanılır. Çünkü bu bağlantılar, az sayıda
# olan sınıfın çevresindeki sınıf sınırını belirler.
# Bu bağlantılara dahil olan çoğunluk sınıfına ait örnekler, veri kümesinden çıkarılır. Bu sayede, az sayıda olan
# sınıfın örneklerinin korunması sağlanırken, çoğunluk sınıfından gereksiz örneklerin azaltılması hedeflenir.
# Sonuç olarak, Tomek Links for Undersampling yöntemi, az sayıda olan sınıfın sınırlarını netleştirerek veri kümesinin
# dengesizliğini azaltırken, aynı zamanda çoğunluk sınıfından gereksiz örnekleri çıkarmayı amaçlar.

y = df["Target"]
X = df.drop("Target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.under_sampling import TomekLinks

# summarize class distribution
counter = Counter(y)
print(counter) # {0: 8500, 1: 1627}
# define the undersampling method
undersample = TomekLinks()
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter) # {0: 8344, 1: 1627}


y = df["Target"]
X = df.drop("Target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Edited Nearest Neighbors Rule for Undersampling
from imblearn.under_sampling import EditedNearestNeighbours
# summarize class distribution
counter = Counter(y)
print(counter) # {0: 8500, 1: 1627}
# define the undersampling method
undersample = EditedNearestNeighbours(n_neighbors=3)
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter) # {0: 7476, 1: 1627}

# Random Undersampling
y = df["Target"]
X = df.drop("Target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.under_sampling import RandomUnderSampler
# summarize class distribution
print(Counter(y)) # {0: 8500, 1: 1627}
# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
X_over, y_over = undersample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_over)) # {0: 1627, 1: 1627}



# Combinations of Keep and Delete Methods

# One-Sided Selection for Undersampling

# One-Sided Selection for Undersampling (Tek Taraflı Seçim Azaltma) yöntemi, az sayıda olan sınıfın örneklerini korurken,
# çoğunluk sınıfından gereksiz örnekleri çıkarmayı hedefleyen bir tekniktir. Bu yöntem, sınıf dengesizliğini azaltmak
# için kullanılır.
# İşleyişi şu şekildedir:
# Başlangıçta, veri kümesindeki her bir az sayıda olan sınıfa ait örnek için, bu örneğe en yakın komşuları belirlenir.
# Bu işlem sonucunda, az sayıda olan sınıfın örnekleri ve bunların en yakın komşuları arasındaki ilişki incelenir. Eğer
# bir az sayıda olan sınıfın örneği ve en yakın komşusu aynı sınıfa ait değilse, yani farklı sınıflara aitlerse, bu
# örnekler "bilgi kaybı" olarak değerlendirilir ve çoğunluk sınıfından gereksiz örnekler olarak işaretlenir.
# Daha sonra, belirlenen bu gereksiz örnekler ile az sayıda olan sınıfın örnekleri birlikte göz önünde bulundurularak,
# daha dengeli bir veri kümesi oluşturulmaya çalışılır. Bunun için, gereksiz örnekler çoğunluk sınıfından çıkarılır ve
# az sayıda olan sınıfın örnekleri korunur.
# Sonuç olarak, One-Sided Selection for Undersampling yöntemi, az sayıda olan sınıfın örneklerini korurken, çoğunluk
# sınıfından gereksiz örnekleri çıkararak veri kümesinin dengesizliğini azaltır. Bu sayede, daha dengeli bir veri
# kümesi elde edilir ve sınıflandırma performansı artırılabilir.

y = df["Target"]
X = df.drop("Target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.under_sampling import OneSidedSelection

# summarize class distribution
counter = Counter(y)
print(counter) # {0: 8500, 1: 1627}
# define the undersampling method
undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter) # {0: 3702, 1: 1627}

# Neighborhood Cleaning Rule for Undersampling

# "Neighborhood Cleaning Rule for Undersampling" (Komşuluk Temizleme Kuralı Azaltma), az sayıda olan sınıfın
# örneklerini korurken, çoğunluk sınıfından gereksiz örnekleri temizlemek için kullanılan bir yöntemdir.
# İşleyişi şu şekildedir:
# Başlangıçta, her bir örneğin etrafındaki komşuları incelenir.
# Eğer bir örnek, etrafındaki komşularının tamamı farklı sınıflara aitse, yani homojen değilse, bu örnek "bilgi kaybı"
# olarak değerlendirilir ve çoğunluk sınıfından gereksiz örnekler olarak işaretlenir.
# Daha sonra, belirlenen bu gereksiz örnekler ile az sayıda olan sınıfın örnekleri birlikte göz önünde bulundurularak,
# daha dengeli bir veri kümesi oluşturulmaya çalışılır. Bunun için, gereksiz örnekler çoğunluk sınıfından çıkarılır
# ve az sayıda olan sınıfın örnekleri korunur.
# Sonuç olarak, Neighborhood Cleaning Rule for Undersampling yöntemi, az sayıda olan sınıfın örneklerini korurken,
# çoğunluk sınıfından gereksiz örnekleri çıkararak veri kümesinin dengesizliğini azaltır. Bu sayede, daha dengeli
# bir veri kümesi elde edilir ve sınıflandırma performansı artırılabilir.

y = df["Target"]
X = df.drop("Target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.under_sampling import NeighbourhoodCleaningRule
# summarize class distribution
counter = Counter(y)
print(counter) # {0: 8500, 1: 1627}
# define the undersampling method
undersample = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter) # {0: 7190, 1: 1627}


# RFE
df.head()