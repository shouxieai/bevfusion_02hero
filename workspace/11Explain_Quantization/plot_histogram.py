import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)
data = np.random.randn(1000)
plt.hist(data, bins=100)

plt.title('Histogram')
plt.xlabel('Vaule')
plt.ylabel('Frequency')
plt.show()