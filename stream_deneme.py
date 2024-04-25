from streamlit import pyplot, title, write

# Başlık
title('Merhaba, PowerPuff Girls!')

# Metin
write('Bu bir deneme uygulamasıdır.')

# Grafik
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
pyplot(plt)

