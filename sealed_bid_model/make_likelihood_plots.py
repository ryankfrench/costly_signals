#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

for n in range(2,6):
    x = np.linspace(0,1000,101)
    like = scipy.stats.beta.pdf(x / 1000, n - 1, 2) / 1000
    plt.plot(x, like, label=str(n)+" signals")
plt.xlabel('Range')
plt.title('Probability Density')
plt.legend()
plt.savefig('range_pdf.png')

plt.cla()
for n in range(2,6):
    x = np.linspace(0,1000,101)
    like = scipy.stats.beta.pdf((1000-x) / 1000, n - 1, 2) / 1000
    plt.plot(x, like, label=str(n)+" signals")
plt.xlabel('Uncertainty')
plt.title('Probability Density')
plt.legend()
plt.savefig('uncertainty_pdf.png')

