"""
Dataliner's linki -- alttaki da kodu:
https://github.com/emreyldzgl/Passenger-Satisfaction-Forecasting-Streamlit-APP/blob/main/pages/single.py
"""

"""
import joblib
import streamlit as st
from function import *

# Genel Sayfa Ayarları
st.set_page_config(layout="centered", page_title="Dataliners Hava Yolları",
                   page_icon="images/airplane.ico")

st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.image("./images/b7.png")

# Background Resminin Ayarlanması
img = get_img_as_base64("./images/Fearless - 7.jpeg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image:url("data:image/png;base64,{img}");
background-size: cover;
background-position: center top;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"]
{{background: rgba(56,97,142,0.3);}}
{{[data-testid="stVerticalBlockBorderWrapper"]
{{background-color: rgba(38, 38, 54, 0.3); border-radius: 16px;}}
.st-ds
{{background-color: rgba(38, 39, 48, 0);}}
[.data-testid="stColorBlock"]
{{background-color: rgba(38, 39, 10;}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(
    f"""
    <style>
        section[data-testid="stSidebar"] {{
            width: 200px !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sayfa Başlığı ve Yazı Stili
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-family: Yellow peace;
            font-weight: lighter;
            color: rgba(91, 162, 194);
            font-size: 2.5rem;
            padding-bottom: 20px;
        }
        .me {
            text-align: center;
            font-family: Yellow peace;
            color: rgba(94, 78, 207);
            font-size: 1 rem;
            padding: 0;
            margin: 0;
        }
        .a {
            text-align: center;
            font-family: Yellow peace;
            color: rgba(94, 78, 207);
            padding: 0;
            margin: 0;
        }

    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='title'> Miuul Airlines R&D </h1>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .st-ec {
        background: rgb(41,72,149,0.3);
    }

    </style>
    """,
    unsafe_allow_html=True
)



# Sayfa Düzenine Tabların Eklenmesi
taba ,tab1, tab2, tab3, tab4 = st.tabs([" ","✈️ Basic Flight Information",
                                                          "👨🏻‍✈️ Airborne Hospitality",
                                                          "👷🏻‍♂️ Operational Service",
                                                          "🧑🏻‍💻 Suitability"])

st.markdown("""
    <style>
    div[role="tablist"] {
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)


# Ana Ekran Giriş Sayfası
taba.image("./images/Fearless - 2.png")


# Kullanıcı Input Girişleri Ekran Tasarımı

gendera = {" Male 👦🏻 ": "Male",
           " Female 👩🏻‍🦰": "Female"}

Customer_Typea = {"Loyal Customer 🌟": "Loyal Customer",
                  "Disloyal Customer": "disloyal Customer"}

Type_of_Travela = {"Business Travel  💼": "Business travel",
                   "Personal Travel  🏖️": "Personal Travel"}

Classa = {"Business": "Business",
          "Eco +": "Eco Plus",
          "Eco": "Eco"}

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        Gender = st.radio("Gender", list(gendera.keys()),
                          help="Passenger Sex", horizontal=True)
        Customer_Type = st.radio("Customer Type", list(Customer_Typea.keys()), help="Passenger Type", horizontal=True)
        Age = st.number_input("Your Age", 0, 120, help="The actual age of the passenger", step=1)
    if Age < 7:
        Age = 7
    elif Age > 85:
        Age = 85

    with col2:
        Type_of_Travel = st.radio("Travel Type", list(Type_of_Travela.keys()),
                                  help="The purpose of the passenger's flight", horizontal=True)
        Class = st.radio("Fare Class", list(Classa.keys()), help="Ticket Type", horizontal=True)
        Flight_Distance = st.number_input("Distance", 0, 10000, help="Distance of Flight", step=1)
    if Flight_Distance < 31:
        Flight_Distance = 31
    elif Flight_Distance > 4983:
        Flight_Distance = 4983

# Girilen Input Ara Kayıt
Gender = gendera[Gender]
Customer_Type = Customer_Typea[Customer_Type]
Type_of_Travel = Type_of_Travela[Type_of_Travel]
Class = Classa[Class]

with tab2:
    col3, col4 = st.columns(2)

    with col3:
        Inflight_service = st.slider("Inflight Service 👨🏻‍💼", 0, 5,
                                     help="Level of satisfaction with inflight service", step=1)
        Inflight_entertainment = st.slider("Inflight Entertainment 🎮🎧", 0, 5,
                                           help="Quality of Inflight entertainment system", step=1)
        Inflight_wifi_service = st.slider("Wifi Service ᯤ", 0, 5,
                                          help="Satisfaction level with Wi-Fi service on board", step=1)

    with col4:
        Leg_room_service = st.slider("Leg Room Service 📏", 0, 5, help="Legroom suitability", step=1)
        Seat_comfort = st.slider("Seat Comfort 💺", 0, 5, help="Seat comfort level", step=1)
        Food_and_drink = st.slider("Food & Drink 🥪🧋", 0, 5, help="Food and drink flavor level", step=1)

with tab3:
    col5, col6 = st.columns(2)

    Baggage_handling = tab3.slider("Baggage Handling 🧳", 1, 5,
                                   help="Baggage handling", step=1)

    col7, col8 = st.columns(2)

    with col5:
        Departure_Arrival_time_convenient = st.slider("Departure Arrival Time 🕙📅", 0, 5,
                                                      help="Departure/arrival time convenient level", step=1)
        Cleanliness = st.slider("Cleanliness ✨", 0, 5,
                                help="Airplane cleanliness", step=1)

    with col7:
        Departure_Delay_in_Minutes = st.number_input("Departure Delay in Minutes 🛫", 0, 10000,
                                                     help="Departure delay.", step=1)
    if Departure_Delay_in_Minutes < 0:
        Departure_Delay_in_Minutes = 0
    elif Departure_Delay_in_Minutes > 1128:
        Departure_Delay_in_Minutes = 1128

    with col6:
        Gate_location = st.slider("Gate Location 🏛️", 0, 5,
                                  help="Level of availible with the gate location", step=1)
        On_board_service = st.slider("On Board Service ✅", 0, 5,
                                     help="Flight satisfaction with on-board service", step=1)

    with col8:
        Arrival_Delay_in_Minutes = st.number_input("Arrival Delay in Minutes 🛬", 0, 10000,
                                                   help="Arrival delay.", step=1)
    if Arrival_Delay_in_Minutes < 0:
        Arrival_Delay_in_Minutes = 0
    elif Arrival_Delay_in_Minutes > 1115:
        Arrival_Delay_in_Minutes = 1115

with tab4:
    Ease_of_Online_booking = st.slider("Ease of Online Booking 🔍", 0, 5, help="Easy online booking rate", step=1)
    Online_boarding = st.slider("Online Boarding 🎟", 0, 5, help="User friendly online boarding", step=1)
    Checkin_service = st.slider("Check-in Service 🙋🏻‍♂️", 0, 5, help="User friendly checkin service", step=1)

    col9, col10, col11, col12 = st.columns(4)


# Girilen Inputların Kaydı ve Model Sonuçları
with tab4:
    if col9.button("PREDICT"):

        fd = pd.DataFrame({'Gender': [Gender],
                           'Customer Type': [Customer_Type],
                           'Age': [Age],
                           'Type of Travel': [Type_of_Travel],
                           'Class': [Class],
                           'Flight Distance': [Flight_Distance],
                           'Inflight wifi service': [Inflight_wifi_service],
                           'Departure/Arrival time convenient': [Departure_Arrival_time_convenient],
                           'Ease of Online booking': [Ease_of_Online_booking],
                           'Gate location': [Gate_location],
                           'Food and drink': [Food_and_drink],
                           'Online boarding': [Online_boarding],
                           'Seat comfort': [Seat_comfort],
                           'Inflight entertainment': [Inflight_entertainment],
                           'On-board service': [On_board_service],
                           'Leg room service': [Leg_room_service],
                           'Baggage handling': [Baggage_handling],
                           'Checkin service': [Checkin_service],
                           'Inflight service': [Inflight_service],
                           'Cleanliness': [Cleanliness],
                           'Departure Delay in Minutes': [Departure_Delay_in_Minutes],
                           'Arrival Delay in Minutes': [Arrival_Delay_in_Minutes]})

        df = pd.read_csv("data/data.csv")

        input_df = pred_data(df, fd)

        new_model = joblib.load("model/lgbm.pkl")
        pred = new_model.predict(input_df)

        if pred[0] == 0:
            with tab4:
                tab4.image("images/dissatisfied.png")

        else:
            with tab4:
                tab4.image("images/satisfied.png")

# Sayfa Footer HTML Kod Uygulaması
with open("style/footer.html", "r", encoding="utf-8") as pred:
    footer_html = f"""{pred.read()}"""
    st.markdown(footer_html, unsafe_allow_html=True)
"""