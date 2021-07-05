import matplotlib.pyplot as plt
import autograd.numpy as np
import os
import sys
syspath = os.path.dirname(os.path.realpath(__file__)) + "/.."
sys.path.insert(0, syspath)

from vbsw_module.data_generation.samplers import *

from vbsw_module.functions.basic_functions import *
from vbsw_module.functions.df import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

unif = (35/255,87/255,137/255)
ada = (217/255,72/255,1/255)

#%% tanh
boundaries = np.array([[0], [1]])
x = np.linspace(0, 1, 1000)
y = tanh(x)
d2 = taylor_w(tanh, 2, x, 5e-2)


x_ada = tbs(boundaries, np.reshape(x, (1000,1)), [tanh], 100, 5e-2, 100, 10)
y_ada = tanh(x_ada)

fig, ax1 = plt.subplots()

color = 'black'
ax1.set_ylabel(r'f(x)', color=color)
ax1.set_xlabel(r'x', color=color)
ax1.plot(x, y, color=color)
ax1.plot(x_ada, y_ada, '+', color = ada)
ax1.plot(x_ada, np.zeros(100), '+', color = ada)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = ada
ax2.set_ylabel(r'$Df^2_{\epsilon}(x)$', color=color)  # we already handled the x-label with ax1
ax2.plot(x, d2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

for a in [ax1, ax2]:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    a.spines["left"].set_visible(False)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(syspath + '/figures/tbs/tanh-sample', bbox_inches='tight')
plt.close()

#%% runge

boundaries = np.array([[0], [1]])
x = np.linspace(0, 1, 1000)
y = runge(x)
d2 = taylor_w(runge, 2, x, 5e-2)

x_ada = tbs(boundaries, np.reshape(x, (1000,1)), [runge], 100, 5e-2, 100, 2)
y_ada = runge(x_ada)

fig, ax1 = plt.subplots()

color = 'black'
ax1.set_ylabel(r'f(x)', color=color)
ax1.set_xlabel(r'x', color=color)
ax1.plot(x, y, color=color)
ax1.plot(x_ada, y_ada, '+', color = ada)
ax1.plot(x_ada, np.zeros(100) + y[0], '+', color = ada)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = ada
ax2.set_ylabel(r'$Df^2_{\epsilon}(x)$', color=color)  # we already handled the x-label with ax1
ax2.plot(x, d2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

for a in [ax1, ax2]:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    a.spines["left"].set_visible(False)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(syspath + '/figures/tbs/runge-sample', bbox_inches='tight')
plt.close()


