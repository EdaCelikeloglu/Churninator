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

def process_data(df):
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


    # LOF
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(df[num_cols])

    df_scores = clf.negative_outlier_factor_

    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked=True, xlim=[0, 100], style='.-')
    plt.show()

    th = np.sort(df_scores)[25]

    outliers = df_scores < th
    #df = df[~outliers]


    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    # Missing values
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
    df["Days_Inactive_Last_Year"].value_counts()


    df["Days_Inactive_Last_Year"].replace(0, 30, inplace=True)
    df["Days_Inactive_Last_Year"].replace(180, 150, inplace=True)


    # RFM
    df["RecencyScore"] = df["Days_Inactive_Last_Year"].apply(lambda x: 5 if x == 30 else
                                                            4 if x == 60 else
                                                            3 if x == 90 else
                                                            2 if x == 120 else
                                                            1 if x == 150 else x)

    df["MonetaryScore"] = pd.qcut(df["Total_Trans_Amt"], 5, labels=[1, 2, 3, 4, 5])
    df["FrequencyScore"] = pd.qcut(df["Total_Trans_Ct"], 5, labels=[1, 2, 3, 4, 5])

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


    # k-means ile müşteri segmentasyonu öncesi standartlaştırmayı yapmak gerek
    # Min-Max ölçeklendirme
    from sklearn.preprocessing import MinMaxScaler


    sc = MinMaxScaler((0,1))
    df[['Days_Inactive_Last_Year','Total_Trans_Ct', 'Total_Trans_Amt']] = sc.fit_transform(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])


    from sklearn.cluster import KMeans
    # model fit edildi.
    kmeans = KMeans(n_clusters = 4, max_iter=50)
    kmeans.fit(df[['Days_Inactive_Last_Year','Total_Trans_Ct', 'Total_Trans_Amt']])

    df["cluster_no"] = kmeans.labels_
    df["cluster_no"] = df["cluster_no"] + 1


    ssd = []

    K = range(1,30)

    for k in K:
        kmeans = KMeans(n_clusters=k).fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])
        ssd.append(kmeans.inertia_) #inertia her bir k değeri için ssd değerini bulur.


    # Optimum küme sayısını belirleme
    from yellowbrick.cluster import KElbowVisualizer
    kmeans = KMeans()
    elbow = KElbowVisualizer(kmeans, k=(2, 20))
    elbow.fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])
    elbow.show()
    elbow.elbow_value_


    # yeni optimum küme sayısı ile model fit edilmiştir.
    kmeans = KMeans(n_clusters = elbow.elbow_value_).fit(df[['Days_Inactive_Last_Year', 'Total_Trans_Ct', 'Total_Trans_Amt']])


    # Cluster_no 0'dan başlamaktadır. Bunun için 1 eklenmiştir.
    df["cluster_no"] = kmeans.labels_
    df["cluster_no"] = df["cluster_no"] + 1



    combine_categories(df, 'Customer_Age_Category', 'Marital_Status', 'Age_&_Marital')
    combine_categories(df, 'Gender', 'Customer_Age_Category', 'Gender_&_Age')
    combine_categories(df, "Card_Category", "Customer_Age_Category", "Card_&_Age")
    combine_categories(df, "Gender", "FrequencyScore", "Gender_&_Frequency")
    combine_categories(df, "Gender", "MonetaryScore", "Gender_&_Monetary")

    df['Total_Amt_Increased'] = np.where((df['Total_Amt_Chng_Q4_Q1'] >= 0) & (df['Total_Amt_Chng_Q4_Q1'] < 1), 0, 1)
    df['Total_Ct_Increased'] = np.where((df['Total_Ct_Chng_Q4_Q1'] >= 0) & (df['Total_Ct_Chng_Q4_Q1'] < 1), 0, 1)


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


    # Personalar
    df["Affluent_criteria"] = (df['Income_Category'] == '$120K +').astype('Int64')
    df["Budget_criteria"] = ((df['Income_Category'] == 'Less than $40K') & (df['Education_Level'].isin(['High School', 'College']))).astype('Int64')
    df["Young_prof_criteria"] = ((df['Customer_Age'] <= 30) & (df['Education_Level'].isin(['College', 'Graduate']))).astype('Int64')
    df["Family_criteria"] = ((df["Marital_Status"].isin(["Married", "Divorced", "Unknown"])) & (df['Dependent_count'] >= 3)).astype(int)
    df["Credit_builder_criteria"] = (df['Credit_Limit'] < 2500).astype(int)  # This threshold is chosen to include individuals with credit limits in the lower percentiles of the distribution, which may indicate a need for credit-building strategies or entry-level credit products.
    df["Digital_criteria"] = (df['Contacts_Count_12_mon'] == 0).astype(int)
    df["High_net_worth_individual"] = ((df['Income_Category'] == '$120K +') & (df['Total_Trans_Amt'] > 5000)).astype('Int64')
    df["Rewards_maximizer"] = ((df['Total_Trans_Amt'] > 10000) & (df['Total_Revolving_Bal'] == 0)).astype(int) # For the Rewards_maximizer column, the threshold for Total_Trans_Amt is also set at $10000. Since rewards maximizers are individuals who strategically maximize rewards and benefits from credit card usage, it's reasonable to expect that they engage in higher levels of spending. Therefore, the threshold of $10000 for Total_Trans_Amt appears appropriate for identifying rewards maximizers, considering that it captures individuals with relatively high spending habits.
    df["May_marry"] = ((df["Age_&_Marital"] == "Young_Single") & (df['Dependent_count'] == 0)).astype(int)


    df["Product_by_Year"] = df["Total_Relationship_Count"] / (df["Months_on_book"] / 12)
    df['Year_on_book'] = df['Months_on_book'] // 12

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # Encoding:
    # Rare encoding:
    df["Card_Category"] = df["Card_Category"].apply(lambda x: "Gold_Platinum" if x == "Platinum" or x == "Gold" else x)
    df["Months_Inactive_12_mon"] = df["Months_Inactive_12_mon"].apply(lambda x: 1 if x == 0 else x)
    df["Ct_vs_Amt"] = df["Ct_vs_Amt"].apply(lambda x: "Dec_ct_inc_amt" if x == "Dec_ct_same_amt" else x)
    df["Ct_vs_Amt"] = df["Ct_vs_Amt"].apply(lambda x: "Inc_ct_inc_amt" if x == "Same_ct_inc_amt" else x)
    df["Contacts_Count_12_mon"] = df["Contacts_Count_12_mon"].apply(lambda x: 5 if x == 6 else x)
    df["Card_&_Age"] = df["Card_&_Age"].apply(lambda x: "Rare" if df["Card_&_Age"].value_counts()[x] < 30 else x)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # Ordinal encoding:
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

    df.head()
    cat_cols, num_cols, cat_but_car = grab_col_names(df)



    df_ = one_hot_encoder(df, ["Gender"], drop_first=True)
    df_.rename(columns={"Gender_M": "Gender"}, inplace=True)

    # KNN Imputer
    numeric_columns = df_.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = [col for col in df_.columns if col not in numeric_columns]
    df_numeric = df_[numeric_columns]
    imputer = KNNImputer(n_neighbors=10)
    df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_columns)
    df_concat = pd.concat([df_numeric_imputed, df[categorical_columns]], axis=1)
    df_concat["Education_Level"] = df_concat["Education_Level"].round().astype('Int64')
    df_concat["Income_Category"] = df_concat["Income_Category"].round().astype('Int64')
    return df_concat[~outliers]

if __name__ == '__main__':
    # Bu kısım yalnızca main.py doğrudan çalıştırıldığında çalışacak.
    df = "Veri yükleniyor..."
    processed_data = process_data(df)
    print("İşlenmiş veriler: ", processed_data)
    # Diğer script kodları...


