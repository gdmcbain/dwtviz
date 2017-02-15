import pywt
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.colors as col
import matplotlib.colorbar as colbar
from itertools import chain

def dwtviz(signal, wavelet='db1', cmap_name='Blues'):
    coefs = pywt.wavedec(signal, wavelet)[1:]  # drop appox coef
    f, ax = plt.subplots(2)
    f.subplots_adjust(hspace=0.025)

    dwt_heatmap(coefs, ax[0], cmap_name)
    ax[0].set_title('wavelet coefficient heatmap')
    ax[1].plot(signal)
    ax[1].set_xlim([0, len(signal)])
    ax[1].set_title('signal', y = -0.15)
    return f

def dwt_heatmap(coefs, ax, cmap_name):
    ax.set_xticks([])

    ax.set_yticks([(i / 9) - (1 / 18) for i in range(1, len(coefs))])
    ax.set_yticklabels(reversed(range(len(coefs))))
    ax.set_ylabel('levels')

    norm = col.Normalize(vmin=min(chain(*coefs)), vmax=max(chain(*coefs)))
    cmap = plt.get_cmap(cmap_name)

    colbar_axis = colbar.make_axes(ax.figure.axes, 'right')
    colbar.ColorbarBase(colbar_axis[0], cmap, norm)

    height = 1 / len(coefs)
    for level, coef_level in enumerate(coefs):
        width = 1 / len(coef_level)
        for n, coef in enumerate(coef_level):
            bottom_left = (0 + (n * width), 1 - ((level + 1) * height))
            color = cmap(norm(coef))
            heat_square = pat.Rectangle(bottom_left, width, height, color=color)
            ax.add_patch(heat_square)

