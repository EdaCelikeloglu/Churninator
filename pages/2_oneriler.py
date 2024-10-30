import streamlit as st
import pandas as pd


st.set_page_config(page_title="Churninator | Öneriler", page_icon=":robot:", layout="centered")


st.markdown("# Öneriler")
st.sidebar.header("Öneriler")

# bu datayı daha önce mainde segment_counts_one olarak yakalamıştık şimdi manuel kaydediyoruz.
data = {
    'Segment': ['About to Sleep', 'Potential Loyalists', 'Promising', 'Hibernating', 'Need Attention',
                'New Customers', 'Loyal Customers', 'At Risk', 'Champions', "Can't Lose"],
    'Count': [124, 45, 38, 32, 14, 11, 11, 2, 1, 1]
}

# DataFrame oluşturma
segment_df = pd.DataFrame(data)
percentages = [124, 45, 38, 32, 14, 11, 11, 2, 1, 1]
total = 279
formatted_percentages = [f"{(count / total * 100):.1f}%" for count in percentages]

# Creating the formatted string for streamlit
formatted_string = f"""
**Bu gruplar**:
- `Uyumak Üzere`: {formatted_percentages[0]}
- `Muhtemel Sadık`: {formatted_percentages[1]}
- `Değerli Olabilir`: {formatted_percentages[2]}
- `Uykuda`: {formatted_percentages[3]}
- `İlgiye İhtiyacı Var`: {formatted_percentages[4]}
- `Yeni Müşteri`: {formatted_percentages[5]}
- `Sadık Müşteri`: {formatted_percentages[6]}
- `Riskli`: {formatted_percentages[7]}
- `Şampiyon`: {formatted_percentages[8]}
- `Çok Kıymetli`: {formatted_percentages[9]}
"""

st.write("Kaybetme riski ile karşı karşıya olduğunuz müşterilerinizi on farklı gruba ayırdık.")
st.write(formatted_string)
st.write("""
   
    
     
     
""")
st.write("**Önerilerimiz:**")


#st.write("`Yüksek Kazanan`, `Birikim Yapan`, `Kredi Notu Odaklı`, `Genç Profesyonel`, `Aile Sahibi`, `Dijital Müşteri`, `Varlıklı`")
#st.write("`Uykuda`, `Riskli`, `Çok Değerli`, `Uyumak Üzere`, `İlgiye İhtiyacı Var`, `Sadık Müşteri`, `Değerli Olabilir`, `Yeni Müşteri`, `Muhtemel Sadık`, `Şampiyon`")

with st.expander("***Ek ürün teşvikleri***"):
    st.markdown("`Yüksek Kazanan`  `Birikim Yapan`  `Kredi Notu Odaklı`  `Genç Profesyonel`  `Aile Sahibi`  `Dijital Müşteri`  `Varlıklı`  `Evlilik Dönemi`  `Düşük Ürün Adedi`")
    st.markdown("**Teklif:** Ek hesaplar açmak veya yeni hizmetlere kaydolmak için Hoşgeldin Faizi gibi indirimli faiz oranları sunun.")
    st.markdown("**Teklif:** Ek hesaplar açmak veya yeni hizmetlere kaydolmak için nakit geri ödeme ödülleri sunun.")

with st.expander("***Finansal Sağlık Kontrolü***"):
    st.markdown("`Uykuda`  `Uyumak Üzere`  `Kredi Notu Odaklı`")
    st.markdown("Özel finansal danışmanlarla kişiselleştirilmiş finansal danışmanlık sağlayın.")
    st.markdown("**Teklif:** Müşterilerin mevcut finansal durumlarını gözden geçirmek, iyileştirilecek alanları belirlemek ve finansal hedeflerine ulaşmak için kişiselleştirilmiş bir plan oluşturmak üzere ücretsiz bir oturum sunun.")

with st.expander("***'Tekrar Hoş Geldiniz' kampanyası***"):
    st.markdown("`Uykuda`  `Düşük Ürün Adedi`  `Uyumak Üzere`")
    st.markdown("Hesaplarını yeniden aktive eden müşterilere sınırlı süreli bir promosyon sunun. Müşterileri özel tekliflerden yararlanmaları için hedeflenmiş e-postalar veya doğrudan postalama göndererek yeniden aktifleştirme sürecine teşvik edin.")
    st.markdown("**Aksiyon:** Kredi kartlarında nakit geri ödeme ödülü veya yıllık ücret muafiyeti sunun.")

with st.expander("***Aile Paketi***"):
    st.markdown("`Aile Sahibi`  `Evlilik Dönemi`")
    st.markdown("Aylık bakım ücreti olmayan bir kontrol hesabı, rekabetçi faiz oranlarına sahip bir tasarruf hesabı ve üniversite tasarruf planında indirimli bir oran içeren bir 'Aile Paketi' oluşturun.")
    st.markdown("**Teklif:** Aylık masrafları için otomatik ödemeleri ayarlayan ailelere bir nakit bonus veya hediye kartı sunun.")

with st.expander("***'Kariyer Başlangıcı' Paketi***"):
    st.markdown("`Genç Profesyonel`")
    st.markdown("Bütçeleme araçlarına sahip bir kontrol hesabı, öğrenci kredisi refinansmanı seçeneği ve kariyer geliştirme kaynaklarına erişim sunan bir paket.")
    st.markdown("**Teklif:** 'Kariyer Başlangıcı' paketine kaydolan genç profesyoneller için ağ kurma etkinlikleri veya atölyeler düzenleyin.")

with st.expander("***Elit Ödüller Programı***"):
    st.markdown("`Şampiyon`  `Çok Değerli`")
    st.markdown("Üst düzey müşteriler için VIP müşteri hizmetleri, premium seyahat ayrıcalıkları ve artırılmış nakit iade oranları gibi özel ödüller ve avantajlar sunun.")
    st.markdown("**Teklif:** Üst düzey müşterilere kişiselleştirilmiş teşekkür notları veya sürpriz hediyeler göndererek sadakatlerini daha da pekiştirin.")
    st.markdown("**Teklif:** Şampiyonlar segmentindeki müşterileri elit ödüller programına katılmaya davet edin ve üyeliğin lüks ve prestijle ilişkilendirildiğini gösterin.")

with st.expander("***Mevsimsel Harcama Takibi***"):
    st.markdown("`Harcama Miktarı Azalan`  `Harcama Adedi Azalan`")
    st.markdown("Müşterilerin finansal davranışlarını anlamalarına ve harcama alışkanlıklarını optimize etme fırsatlarını belirlemelerine yardımcı olun, böylece sadakati artırın.")
    st.markdown("**Teklif:** Müşterilere harcama alışkanlıkları ve çeyrekten çeyreğe değişimler konusunda kişiselleştirilmiş bilgiler gönderin.")

st.markdown("""


""")


st.markdown("<center>© 2024  DEG Bilgi Teknolojileri Danışmanlık ve Dağıtım A.Ş. Tüm hakları saklıdır.</center>", unsafe_allow_html=True)