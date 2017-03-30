# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 20:12:12 2017

@author: holyland
"""

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

#标准正态分布的公式是 exp(-x**2/2)/sqrt(2*pi)
#而norm.pdf(x) =  exp(-x**2/2)/sqrt(2*pi)

#连续随机变量对象都有如下方法：

#rvs：对随机变量进行随机取值，可以通过size参数指定输出的数组的大小。
#pdf：随机变量的概率密度函数。
#cdf：随机变量的累积分布函数，它是概率密度函数的积分。
#sf：随机变量的生存函数，它的值是1-cdf(t)。
#ppf：累积分布函数的反函数。
#stat：计算随机变量的期望值和方差。
#fit：对一组随机取样进行拟合，找出最适合取样数据的概率密度函数的系数。

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

# Calculate a few first moments:

mean, var, skew, kurt = norm.stats(moments='mvsk')

# Display the probability density function (``pdf``):
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)#linspace等差数列,ppf为反函数
ax.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')

# Alternatively, the distribution object can be called (as a function)
# to fix the shape, location and scale parameters. This returns a "frozen"
# RV object holding the given parameters fixed.

# Freeze the distribution and display the frozen ``pdf``:

rv = norm()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

# Check accuracy of ``cdf`` and ``ppf``:

vals = norm.ppf([0.001, 0.5, 0.999])
np.allclose([0.001, 0.5, 0.999], norm.cdf(vals))
# True

# Generate random numbers:

r = norm.rvs(size=1000)

# And compare the histogram:

ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()