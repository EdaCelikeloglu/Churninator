#######################
# Recommendation Engine
#######################

# Churn edeceğini gördüğümüz müşteriyi tutmaya yönelik olarak banka çalışanına, müşterinin diğer değişkenlerdeki durumuna göre
1.
df.groupby("Total_Relationship_Count")["Target"].mean()
# 1   0.256
# 2   0.278
# 3   0.174
# 4   0.118
# 5   0.120
# 6   0.105
# TODO churn etmesi beklenen müşteriye, bankanın başka ürünlerinden kampanyalı satış yapmaya çalışmalıyız.


2. df.groupby("Target")["Avg_Utilization_Ratio"].mean() # TODO borcu düşük olanların churn etme oranı daha yüksek (x3.5).
# 0   0.296
# 1   0.162

3. df.groupby("Target")["Total_Revolving_Bal"].mean() # TODO
# 0   1256.604
# 1    672.823

4. df.groupby("Income_Category")["Total_Revolving_Bal"].mean()
df.groupby("Income_Category")["Total_Trans_Amt"].mean()
# TODO 1. Çekilen kredi miktarı da, yapılan harcama miktarı da müşteri gelirlerine kıyasla stabil.
# TODO 2. Bu da, düşük gelirli müşterilerin, Avg_Utilization_Ratio'sunun yani borç ödeme zorluğu oranını artırıyor.
# TODO 3. Borcu olan müşteriler, bankadan ayrılamıyor.
# TODO 4. Müşterileri bankada tutmak için A) ürün sat, B) borcunu artır -- mesela kk limitini artırmayı teklif et.

5. # TODO Şirket mottosu: "Biz borçlunun yanındayız!"


6. df.groupby("Target")["Important_client_score"].mean()
# 0   11.701
# 1    9.863
# TODO Banka, önemli müşterileri tutmakta başarılı!

7. df["Contacts_Count_12_mon"].describe().T
df.groupby("Contacts_Count_12_mon")["Target"].mean() # 6'ların hepsi churn. Yükseldikçe churn olasılığı artıyor.
# TODO Number of contacts with the bank might indicate dissatisfaction or queries.
# 0   0.018
# 1   0.072
# 2   0.125
# 3   0.201
# 4   0.226
# 5   0.335
# 6   1.000

8. # TODO öneri: Total_dependent_count fazla olanlara ek kart öner.

