import streamlit as st

# Başlık
st.title('Merhaba, PowerPuff Girls!')

# Metin
st.write('Bu bir deneme uygulamasıdır.')

# Grafik
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
st.pyplot(plt)

