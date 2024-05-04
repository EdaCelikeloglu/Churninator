import plotly.express as px
import streamlit as st
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

st.set_page_config(page_title="Plotting Demo", page_icon="📈", layout="wide")

st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

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


# Ürün sayısı arttıkça Churn olasılığı azalıyor
st.write("Ürün sayısı arttıkça Churn olasılığı azalıyor.")
mean_target_by_relationship = df.groupby("Total_Relationship_Count")["Target"].mean().reset_index()
fig = px.bar(mean_target_by_relationship, x="Total_Relationship_Count", y="Target",
             labels={"Total_Relationship_Count": "Toplam Ürün Sayısı", "Target": "Target"},
             title="Toplam Ürün Sayısına Göre Target", color_discrete_sequence=["blue"])
st.plotly_chart(fig)

# Yeni gelen müşteriler risk mi?
st.write("Yeni gelen müşteriler risk mi?")
mean_target_by_inactive_months = df.groupby("Months_Inactive_12_mon")["Target"].mean().reset_index()
fig = px.bar(mean_target_by_inactive_months, x="Months_Inactive_12_mon", y="Target",
             labels={"Months_Inactive_12_mon": "İnaktif Ay Sayısı", "Target": "Target"},
             title="İnaktif Ay Sayısına Göre Target", color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig)

# Borcu çok olanlar gidemiyor
mean_utilization_by_target = df.groupby("Target")["Avg_Utilization_Ratio"].mean().reset_index()
mean_revolving_bal_by_target = df.groupby("Target")["Total_Revolving_Bal"].mean().reset_index()
fig_utilization = px.bar(mean_utilization_by_target, x="Target", y="Avg_Utilization_Ratio",
                         labels={"Target": "Hedef", "Avg_Utilization_Ratio": "Borç/Kredi Limiti"},
                         title="Targete Göre Borç/Kredi Limiti", color_discrete_sequence=px.colors.qualitative.Pastel)
fig_utilization.update_layout(height=400, width=400)
fig_revolving_bal = px.bar(mean_revolving_bal_by_target, x="Target", y="Total_Revolving_Bal",
                           labels={"Target": "Hedef", "Total_Revolving_Bal": "Ortalama Devir Bakiyesi"},
                           title="Hedefe Göre Ortalama Devir Bakiyesi", color_discrete_sequence=px.colors.qualitative.Pastel)
fig_revolving_bal.update_layout(height=400, width=400)
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_utilization)
with col2:
    st.plotly_chart(fig_revolving_bal)


# Gelir kategorilerine göre ortalama devir bakiyesi
fig = px.bar(df, x="Income_Category", y="Total_Revolving_Bal",
             labels={"Income_Category": "Gelir Kategorisi", "Total_Revolving_Bal": "Ortalama Devir Bakiyesi"},
             title="Gelir Kategorisine Göre Ortalama Devir Bakiyesi",
             color="Income_Category", color_discrete_sequence=px.colors.qualitative.Pastel)

# Grafiği görüntüle
st.plotly_chart(fig, use_container_width=True)


#bunları sadece aşağıdakileri etkiliyordur diye aldım buraya
df['Total_Amt_Increased'] = np.where((df['Total_Amt_Chng_Q4_Q1'] > 0) & (df['Total_Amt_Chng_Q4_Q1'] < 1), 0, 1)
df["Has_debt"] = np.where((df["Credit_Limit"] > df["Avg_Open_To_Buy"]), 1, 0).astype(int)

#bu grafiğini çizdirmek istediğimiz bir değişkendi
df["Important_client_score"] = df["Total_Relationship_Count"] * (df["Months_on_book"] / 12)

# Target'e göre Important_client_score'un grafiği:
mean_scores_by_target = df.groupby("Target")["Important_client_score"].mean().reset_index()
fig = px.bar(mean_scores_by_target, x="Target", y="Important_client_score",
             labels={"Target": "Hedef", "Important_client_score": "Ortalama Puan"},
             title="Targete Göre Important_client_score")
fig.update_layout(height=400, width=400)
st.plotly_chart(fig)

# müşteri ile iletişim sayısı ve target:
mean_churn_by_contact = df.groupby("Contacts_Count_12_mon")["Target"].mean().reset_index()
mean_churn_by_contact = mean_churn_by_contact.rename(columns={"Target": "Churn_Rate"})
fig = px.bar(mean_churn_by_contact, x="Contacts_Count_12_mon", y="Churn_Rate",
             labels={"Contacts_Count_12_mon": "İletişim Sayısı", "Churn_Rate": "Ortalama Churn Oranı"},
             title="İletişim Sayısına Göre Ortalama Churn Oranı")
fig.update_layout(height=400, width=400)
st.plotly_chart(fig)

# bazı yeni değişkenler:
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

combine_categories(df, 'Customer_Age_Category', 'Marital_Status', 'Age_&_Marital')
combine_categories(df, 'Gender', 'Customer_Age_Category', 'Gender_&_Age')
combine_categories(df, "Card_Category", "Customer_Age_Category", "Card_&_Age")
combine_categories(df, "Gender", "FrequencyScore", "Gender_&_Frequency")
combine_categories(df, "Gender", "MonetaryScore", "Gender_&_Monetary")

df['Total_Amt_Increased'] = np.where((df['Total_Amt_Chng_Q4_Q1'] >= 0) & (df['Total_Amt_Chng_Q4_Q1'] < 1), 0, 1)
df['Total_Ct_Increased'] = np.where((df['Total_Ct_Chng_Q4_Q1'] >= 0) & (df['Total_Ct_Chng_Q4_Q1'] < 1), 0, 1)


# Bunları çok anlamadım sadece buraya ekledim
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


#Age_&_Marital   Gender_&_Age        Card_&_Age değişkenlerini target ile baktım:
fig_age_marital = px.histogram(df, x="Age_&_Marital", color="Target", barmode="group",
                                labels={"Age_&_Marital": "Yaş ve Medeni Durum", "Target": "Churn Durumu"},
                                title="Yaş ve Medeni Duruma Göre Churn Durumu")
fig_age_marital.update_layout(xaxis_title="Yaş ve Medeni Durum", yaxis_title="Sayı")

fig_gender_age = px.histogram(df, x="Gender_&_Age", color="Target", barmode="group",
                               labels={"Gender_&_Age": "Cinsiyet ve Yaş", "Target": "Churn Durumu"},
                               title="Cinsiyet ve Yaşa Göre Churn Durumu")
fig_gender_age.update_layout(xaxis_title="Cinsiyet ve Yaş", yaxis_title="Sayı")

fig_card_age = px.histogram(df, x="Card_&_Age", color="Target", barmode="group",
                             labels={"Card_&_Age": "Kart ve Yaş", "Target": "Churn Durumu"},
                             title="Kart ve Yaşa Göre Churn Durumu")
fig_card_age.update_layout(xaxis_title="Kart ve Yaş", yaxis_title="Sayı")
st.plotly_chart(fig_age_marital)
st.plotly_chart(fig_gender_age)
st.plotly_chart(fig_card_age)


# burada Dec_ct_dec_amt kategorisi nedir? Çok fazla yoğunluk var orda
#  Ct_vs_Amt ile Target:
fig = px.histogram(df, x="Ct_vs_Amt", color="Target", barmode="group",
                   title="Ct_vs_Amt Değişkeninin Target İle İlişkisi",
                   labels={"Ct_vs_Amt": "Ct_vs_Amt", "Target": "Target Ortalaması"},
                   color_discrete_map={0: "lightblue", 1: "salmon"})

fig.update_layout(bargap=0.1)
st.plotly_chart(fig)

# bunun notunu almışım birlikte yorumlayalım.:
fig = px.scatter(df, x="Credit_Limit", y="Total_Revolving_Bal", color="Income_Category",
                 title="Kredi Limiti ve Devir Bakiyesi İlişkisi",
                 labels={"Credit_limit": "Kredi Limiti", "Total_revolving_Bal": "Devir Bakiyesi"},
                 color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(height=800, width=1200)
st.plotly_chart(fig)

#büyük Pasta
#'Education_Level' 'Income_Category' bunları da koycam Nanlar sorun çıkardı
fig = px.sunburst(df, path=['Target', 'Gender', 'Customer_Age_Category', 'Marital_Status'])
fig.update_layout(height=1000, width=1000)
# Streamlit ile gösterme
st.plotly_chart(fig)
#bunun farklı versiyonlarını deneyelim
















progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")