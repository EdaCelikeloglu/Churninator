import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from urllib.error import URLError
import pandas as pd


st.set_page_config(page_title="DataFrame Demo", page_icon="📊")

# Başlık
st.title('Merhaba, PowerPuff Girls!')


#images
im = Image.open("denemeresim.jpg")
st.image(im, width=700, caption="Power")



st.text('Bu bir streamlit text.')
# Metin
st.write('Bu bir streamlit write.')

st.header('This is a header')
st.subheader('This is a subheader')

st.markdown('This is a normal Markdown')
st.markdown('# This is a bold Markdown')
st.markdown('## This is a thin-bold Markdown')
st.markdown('* This is a Markdown with point')
st.markdown('** This is a small bold Markdown **')

st.success('Successful')
st.markdown('`This is a markdown`')
st.info("This is an information")
st.warning('This is a warning')
st.error('This is an error')

st.selectbox('Your Occupation', ['Programmer', 'Data Scientist'])

st.multiselect('Where do you work', ('London','Istanbul','Berlin'))

st.button('Simple Button')

st.slider('What is your level', 0,40, step=5)


#html
html_temp = """

<div style="background-color:tomato;padding:1.5px">

<h1 style="color:white;text-align:center;">Demo Web App </h1>

</div><br>"""

st.markdown(html_temp, unsafe_allow_html=True)

st.title('This is for a good design')

st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)






#Graphs
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)
st.pyplot(fig)

st.title('This is for a good design')

# Grafik
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
st.pyplot(plt)





st.markdown("# DataFrame Demo")
st.sidebar.header("DataFrame Demo")
st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
)


@st.cache_data
def get_UN_data():
    AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
    return df.set_index("Region")


try:
    df = get_UN_data()
    countries = st.multiselect(
        "Choose countries", list(df.index), ["China", "United States of America"]
    )
    if not countries:
        st.error("Please select at least one country.")
    else:
        data = df.loc[countries]
        data /= 1000000.0
        st.write("### Gross Agricultural Production ($B)", data.sort_index())

        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
        )
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                color="Region:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )

