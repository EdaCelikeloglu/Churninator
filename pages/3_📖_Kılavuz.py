import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import seaborn as sns
import warnings


warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

st.set_page_config(page_title="Churninator | Kılavuz", page_icon="🤖", layout="wide")
st.sidebar.header("Kılavuz")

st.markdown("# Churninator ile Geleceğe Hazır Olun!")

st.markdown(
    """
***Churninator***, bankalardaki müşteri churn riskini veri bilimi ile makine öğrenmesinin gücünü kullanarak önceden tahmin eden bir model olarak işletmenizin sadık müşteri tabanını korumanıza yardımcı olurken, müşteri kaybını minimize etmenize olanak tanır. Bu yenilikçi çözüm, rekabet avantajınızı artırır ve işletmenizin karar alma süreçlerini optimize eder.

**Churninator ile Avantajlarınız:**
- **Güvenilir Tahminler:** Churninator, veri bilimi ve makine öğrenmesinin gücünü kullanarak banka müşterilerinizin ayrılma eğilimlerini doğru bir şekilde tahmin eder, böylece proaktif stratejiler oluşturabilirsiniz.
- **Sadık Müşteri Tabanı:** Churninator'un yardımıyla, müşteri kaybını minimize ederek, işletmenizin karlılığını ve uzun vadeli büyüme potansiyelinizi artırabilirsiniz.
- **Rekabet Üstünlüğü:** Veriye dayalı kararlarınızla, pazardaki değişimlere hızla adapte olabilir ve rakiplerinizin önüne geçebilirsiniz.
- **Özelleştirilmiş Çözümler:** Uzman ekibimiz, işletmenizin benzersiz ihtiyaçlarına uygun özelleştirilmiş çözümler sunar, böylece maksimum değer elde edersiniz.
"""
)


st.markdown("# Churninator Nasıl Kullanılır?")

col = st.columns([0.7, 0.3], gap='small')

with col[0]:
    st.write(
        """Hangi kredi kartı müşterilerinizin bankanızdan ayrılacağını tahmin edebilirseniz, onlara nasıl daha iyi hizmet sunabileceğinizi öngörebilir ve bu müşterinizin kararlarını olumlu yönde değiştirebilirsiniz.
    
    
    
    
    
    """
    )
    st.write("Bu örnek veri setinde 10127 müşteri için yaş, maaş, medeni durum, kredi kartı limiti, kredi kartı kategorisi gibi 21 özellik bulunmaktadır.  ")
    st.write("Müşterilerin yalnızca %16'sının ayrıldığını görmekteyiz. Ayrılan ve ayrılmayan müşteriler arasındaki bu dengesizlikten dolayı, ayrılacak müşterileri tahmin etmek için modeli eğitme aşamasında birtakım zorluklar baş gösterse de, ***Churninator*** ile bunların üstesinden gelebilirsiniz.   ")
    st.write(
        """

    """
    )
    st.write(
        "Kaynak: [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers):duck:")

with col[1]:

    # Gülen ve Somurtan Yüz Sembolleri
    smile_image = 'Pages/0.png'
    frown_image = 'Pages/11.png'
    smile_count = 8500
    frown_count = 1627
    total_count = smile_count + frown_count
    total_icons = 100
    grid_size = 20
    smile_icons = round(smile_count / total_count * total_icons)
    frown_icons = total_icons - smile_icons
    icons = [smile_image] * smile_icons + [frown_image] * frown_icons
    for row in range(0, total_icons, grid_size):
        st.image(icons[row:row + grid_size], width=20, caption=None)

st.write("  ")
st.write(
    """

"""
)


df = pd.read_csv("BankChurners.csv")

df.drop([
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"],
    inplace=True, axis=1)

st.write(df.head())

st.markdown(""" Müşteri hakkındaki kişisel bilgiler:
    - **CLIENTNUM:** Müşteri numarası. Tekrar eden değer yok.
    - **Attrition_Flag:** Hedef değişken. Binary. Müşteri churn olmuş ise 1, yoksa 0.
    - **Customer_Age:** Müşterinin yaşı
    - **Gender:** Müşterinin cinsiyeti *(`F`, `M`)*
    - **Dependent_count:** Müşterinin bakmakla yükümlü olduğu kişi sayısı *(`0`, `1`, `2`, `3`, `4`, `5`)*
    - **Education_Level:** Müşterinin eğitim seviyesi *(`Uneducated`, `High School`, `College`, `Graduate`, `Post-Graduate`, `Doctorate`, `Unknown`)*
    - **Marital_Status:** Müşterinin medeni durumu *(`Single`, `Married`, `Divorced`, `Unknown`)*
    - **Income_Category:** Müşterinin hangi gelir kategorisinde olduğu bilgisi *(`Less than $40K`, `$40K - $60K`, `$60K - $80K`, `$80K - $120K`, `$120K+`, `Unknown`)*
   """)
st.markdown(""" Müşterinin bankayla ilişkisi hakkındaki bilgiler:
    - **Card_Category:** Müşterinin sahip olduğu kredi kartı türü *(`Blue`, `Silver`, `Gold`, `Platinum`)*
    - **Months_on_book:** Müşterinin bu bankayla çalıştığı ay sayısı
    - **Total_Relationship_Count:** Müşterinin bankaya ait ürünlerden kaçına sahip olduğu *(`1`, `2`, `3`, `4`, `5`, `6`)*
    - **Months_Inactive_12_mon:** Müşterinin son 12 aylık sürede kredi kartını kullanmadığı ay sayısı
    - **Contacts_Count_12_mon:** Müşteriyle son 12 ayda kurulan iletişim sayısı *(`0`, `1`, `2`, `3`, `4`, `5`, `6`)*
    - **Credit_Limit:** Müşterinin kredi kartı limiti
    - **Total_Revolving_Bal:** Devir bakiyesi. Müşterinin ödemeyi taahhüt ettiği ancak henüz ödenmemiş olan aylık taksitli borç miktarı
    - **Avg_Open_To_Buy:** Müşterinin borç taahhütlerinden sonra arta kalan, harcayabileceği miktar *(`Credit_Limit` - `Total_Revolving_Bal`)*
    - **Avg_Utilization_Ratio:** Müşterinin mevcut kredi kartı borçlarının kredi limitine oranı *(`Total_Revolving_Bal` / `Credit_Limit`)*
    - **Total_Trans_Amt:** Müşterinin son 12 aydaki kredi kartı işlemlerinin tutar toplamı
    - **Total_Amt_Chng_Q4_Q1:** Müşterinin 4. çeyrekteki harcama tutarının, 1. çeyrekteki harcama tutarına kıyasla artış/azalış hareketini gösterir *(`4. Çeyrek` / `1. Çeyrek`)*
    - **Total_Trans_Ct:** Müşterinin son 12 aydaki kredi kartı işlemlerinin adet toplamı
    - **Total_Ct_Chng_Q4_Q1:** Müşterinin 4. çeyrekteki harcama adedinin, 1. çeyrekteki harcama adedine kıyasla artış/azalış hareketini gösterir *(`4. Çeyrek` / `1. Çeyrek`)*
   """)

##########################################################################
# Kategorik değişkenler ve renkler
# Bu grafiklerin kodunu yorum olarak paylaşıyoruz. Daha hızlı yüklenmesi için, Streamlit'te bu grafiklerin ekran görüntüsünü veriyoruz.
categories = ['Gender', "Income_Category", "Education_Level", "Dependent_count", 'Marital_Status', "Card_Category", "Months_Inactive_12_mon", 'Total_Relationship_Count', 'Contacts_Count_12_mon']

filtered_df0 = df[df['Attrition_Flag'] == "Existing Customer"]
filtered_df1 = df[df['Attrition_Flag'] == "Attrited Customer"]

def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"
    return rotation, alignment


def add_labels(angles, values, labels, offset, ax):
    # This is the space between the end of the bar and the label
    padding = 4
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(x=angle, y=value + padding, s=label, ha=alignment, va="center", rotation=rotation, rotation_mode="anchor")


# def circular_bar_graph(df, attrition, col_list, figsize=(10, 10)):
#
#     new_dfs = []  # List to store individual DataFrames
#     for col in col_list:
#         value_counts = df[col].value_counts()
#         new_df = pd.DataFrame({'name': value_counts.index,
#                                'value': (value_counts / len(df[col]) * 100).values,
#                                'group': [col] * len(value_counts)})
#         new_dfs.append(new_df)
#     final = pd.concat(new_dfs, ignore_index=True)
#     VALUES = final["value"].values
#     LABELS = final["name"].values
#     GROUP = final["group"].values
#
#     PAD = 1
#     ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
#     ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
#     WIDTH = (2 * np.pi) / len(ANGLES)
#     OFFSET = np.pi / 2
#
#     unique_groups = []
#     for group in GROUP:
#         if group not in unique_groups:
#             unique_groups.append(group)
#
#     # Calculate the group sizes while maintaining the order
#     GROUPS_SIZE = [len(final[final["group"] == group]) for group in unique_groups]
#
#     # GROUPS_SIZE = [len(i[1]) for i in final.groupby("group")]
#     COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]
#
#     offset = 0
#     IDXS = []
#     for size in GROUPS_SIZE:
#         IDXS += list(range(offset + PAD, offset + size + PAD))
#         offset += size + PAD
#
#     fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
#     ax.set_theta_offset(OFFSET)
#     ax.set_ylim(-50, 100)
#     ax.set_frame_on(False)
#     ax.xaxis.grid(False)
#     ax.yaxis.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     ax.bar(
#         ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, edgecolor="white", linewidth=2)
#
#     add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)
#     ax.text(0, -50, attrition, color='black', ha='center', va='center', fontsize=12)
#
#     # This iterates over the sizes of the groups adding reference lines and annotations.
#     offset = 0
#     #for group, size in zip(final["group"].unique(), GROUPS_SIZE):
#     for group, size in zip(['Gender', "Income Category", "Education Level", "Dependent Count", 'Marital Status', "Card Category", "Months Inactive", 'Relationship Count', 'Contact Count'], GROUPS_SIZE):
#         # Add line below bars
#         x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
#         ax.plot(x1, [-5] * 50, color="#333333")
#
#         # Split the group name if it contains two words
#         group_words = group.split()
#         # Format the group name for display
#         if len(group_words) == 2:
#             group_display = '\n'.join(group_words)  # Display the second word in a new line
#         else:
#             group_display = group  # Keep the group name as it is if it contains only one word
#
#         # Add text to indicate group
#         ax.text(
#             np.mean(x1), -19, group_display, color="#333333", fontsize=8,
#             fontweight="bold", ha="center", va="center")
#
#         # Add reference lines at 20, 40, 60, and 80
#         x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
#         ax.plot(x2, [20] * 50, color="#bebebe", lw=0.8)
#         ax.plot(x2, [40] * 50, color="#bebebe", lw=0.8)
#         ax.plot(x2, [60] * 50, color="#bebebe", lw=0.8)
#         ax.plot(x2, [80] * 50, color="#bebebe", lw=0.8)
#
#         offset += size + PAD
#
#     return fig
#
#
# fig0 = circular_bar_graph(filtered_df0, 0, categories)
# fig1 = circular_bar_graph(filtered_df1, 1, categories)
#
# st.pyplot(fig0)
# st.pyplot(fig1)

##############################picture


import streamlit as st
from PIL import Image

col = st.columns([0.5, 0.5], gap='small')

with col[0]:
    image = Image.open('pic0.png')
    st.image(image, caption='0 Sınıfına Uzaktan Bakış')

with col[1]:
    image_1 = Image.open('pic1.png')
    st.image(image_1, caption='1 Sınıfına Uzaktan Bakış')



###############################picture


##########################################################################

#
# new_dfs = []  # List to store individual DataFrames
# for col in categories:
#     value_counts = filtered_df0[col].value_counts()
#     new_df = pd.DataFrame({'name': value_counts.index,
#                            'value': value_counts.values,
#                            'group': [col] * len(value_counts)})
#     new_dfs.append(new_df)
# final0 = pd.concat(new_dfs, ignore_index=True)
# VALUES = final0["value"].values
# LABELS = final0["name"].values
# GROUP = final0["group"].values
#
# PAD = 3
# ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
# ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
# WIDTH = (2 * np.pi) / len(ANGLES)
# OFFSET = np.pi / 2
#
# GROUPS_SIZE = [len(i[1]) for i in final0.groupby("group")]
#
# offset = 0
# IDXS = []
# for size in GROUPS_SIZE:
#     IDXS += list(range(offset + PAD, offset + size + PAD))
#     offset += size + PAD
#
# fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})
# ax.set_theta_offset(OFFSET)
# ax.set_ylim(-100, 100)
# ax.set_frame_on(False)
# ax.xaxis.grid(False)
# ax.yaxis.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
#
# GROUPS_SIZE = [len(i[1]) for i in final0.groupby("group")]
# COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]
#
# ax.bar(
#     ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS,
#     edgecolor="white", linewidth=2)
#
# add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)
#
# # Extra customization below here --------------------
# # This iterates over the sizes of the groups adding reference lines and annotations.
# offset = 0
# for group, size in zip(final0["group"].unique(), GROUPS_SIZE):
#     # Add line below bars
#     x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
#     ax.plot(x1, [-5] * 50, color="#333333")
#
#     # Add text to indicate group
#     ax.text(
#         np.mean(x1), -20, group, color="#333333", fontsize=14,
#         fontweight="bold", ha="center", va="center")
#
#     # Add reference lines at 20, 40, 60, and 80
#     x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
#     ax.plot(x2, [20] * 50, color="#bebebe", lw=0.8)
#     ax.plot(x2, [40] * 50, color="#bebebe", lw=0.8)
#     ax.plot(x2, [60] * 50, color="#bebebe", lw=0.8)
#     ax.plot(x2, [80] * 50, color="#bebebe", lw=0.8)
#
#     offset += size + PAD
#
# st.pyplot(fig)
#




##########
#
# # Hedef değişken dağılımını pasta grafiği olarak gösterme
# attrition_counts = df["Attrition_Flag"].value_counts()
# colors = px.colors.qualitative.Pastel
# fig_attrition = px.pie(names=attrition_counts.index, values=attrition_counts.values, labels={"Attrition_Flag": "Sayı"},
#                        title="Target Dağılımı", color=attrition_counts.index,
#                        color_discrete_map={attrition: colors[i] for i, attrition in enumerate(attrition_counts.index)})
# fig_attrition.update_layout(height=400, width=400)
# fig_attrition.update_traces(marker=dict(line=dict(color='black', width=1)))  # Kenarlıkları siyah yapma
#
# # Cinsiyet dağılımını çubuk grafikle gösterme
# gender_counts = df["Gender"].value_counts()
# fig_gender = px.bar(x=gender_counts.index, y=gender_counts.values, labels={"x": "Cinsiyet", "y": "Sayı"},
#                     title="Cinsiyet Dağılımı", color=gender_counts.index,
#                     color_discrete_map={gender: colors[i] for i, gender in enumerate(gender_counts.index)})
# fig_gender.update_layout(height=400, width=400)
# fig_gender.update_traces(marker=dict(line=dict(width=1)))
#
# # Grafikleri yan yana gösterme
# col1, col2 = st.columns(2)
# with col1:
#     st.plotly_chart(fig_attrition)
#
# with col2:
#     st.plotly_chart(fig_gender)
#
# # Gelir kategorilerine göre sayıları hesapla
# income_counts = df["Income_Category"].value_counts()
# fig_income = px.bar(x=income_counts.index, y=income_counts.values, labels={"x": "Gelir Kategorisi", "y": "Sayı"},
#                     title="Gelir Kategorisi Dağılımı", color=income_counts.index,
#                     color_discrete_map={income: colors[i] for i, income in enumerate(income_counts.index)})
# fig_income.update_layout(height=400, width=400, showlegend=False)  # Grafik boyutunu ayarla
#
# # Total_Revolving_Bal
# fig_revolving_bal_hist = px.histogram(df, x="Total_Revolving_Bal", title="Toplam Devir Bakiyesi")
# fig_revolving_bal_hist.update_layout(height=400, width=400)  # Grafik boyutunu ayarla
#
# # Grafikleri yan yana gösterme
# col1, col2 = st.columns(2)
# with col1:
#     st.plotly_chart(fig_income)
# with col2:
#     st.plotly_chart(fig_revolving_bal_hist)
#
# # Ürün sayısı
# total_relationship_counts = df["Total_Relationship_Count"].value_counts()
# fig_total_relationship = go.Figure(go.Bar(x=total_relationship_counts.index, y=total_relationship_counts.values,
#                                           marker=dict(color=px.colors.qualitative.Set2)))
#
# fig_total_relationship.update_layout(title="Müşterilerin Ürün Sayısı",
#                                      xaxis_title="Ürün Sayısı",
#                                      yaxis_title="Sayı", height=400, width=400)
#
# # Months_Inactive_12_mon
# months_inactive_counts = df["Months_Inactive_12_mon"].value_counts()
# fig_inactive_months = go.Figure(go.Bar(x=months_inactive_counts.index, y=months_inactive_counts.values,
#                                        marker=dict(color=px.colors.qualitative.Set2)))
#
# fig_inactive_months.update_layout(title="Müşterilerin Son Bir Yılda İnaktif Geçirdiği Ay Sayısı",
#                                   xaxis_title="Inaktif Ay Sayısı",
#                                   yaxis_title="Sayı", height=400, width=400)
#
# # Grafikleri yan yana gösterme
# col1, col2 = st.columns(2)
# with col1:
#     st.plotly_chart(fig_total_relationship)
# with col2:
#     st.plotly_chart(fig_inactive_months)
#
# # Months_on_book
# fig = px.histogram(df, x="Months_on_book", title="Müşterilerin Bankada Geçirdiği Ay Sayısı Dağılımı")
# st.plotly_chart(fig)

st.markdown("<center>© 2024  DEG Bilgi Teknolojileri Danışmanlık ve Dağıtım A.Ş. Tüm hakları saklıdır.</center>", unsafe_allow_html=True)