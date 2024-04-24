import streamlit as st

# Başlık
st.title('Merhaba, Streamlit!')

# Metin
st.write('Bu bir Streamlit uygulamasıdır.')

# Grafik
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
st.pyplot(plt)

#kfhkf