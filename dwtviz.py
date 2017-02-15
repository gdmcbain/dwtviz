import pywt
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.colors as col
import matplotlib.colorbar as colbar
from itertools import chain

def dwtviz(signal, wavelet='db1', level=None, cmap_name='BuPu'):
    coefs = pywt.wavedec(signal, wavelet, level=level)[1:]  # drop appox coef
    f, ax = plt.subplots(2)
    f.subplots_adjust(hspace=0)

    dwt_heatmap(coefs, ax[0], cmap_name)
    ax[1].plot(signal)
    ax[1].set_xlim([0, len(signal)])
    return f

def dwt_heatmap(coefs, ax, cmap_name):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    norm = col.Normalize(vmin=min(chain(*coefs)), vmax=max(chain(*coefs)))
    cmap = plt.get_cmap(cmap_name)

    cbar_axis = ax.figure.add_subplot(222)
    colbar.ColorbarBase(cbar_axis, cmap_name, norm)

    height = 1 / len(coefs)
    for level, coef_level in enumerate(coefs):
        width = 1 / len(coef_level)
        for n, coef in enumerate(coef_level):
            bottom_left = (0 + (n * width), 1 - ((level + 1) * height))
            color = cmap(norm(coef))
            heat_square = pat.Rectangle(bottom_left, width, height, color=color)
            ax.add_patch(heat_square)

