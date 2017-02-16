import collections
import pywt
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.colors as col
import matplotlib.colorbar as colbar
from itertools import chain

def dwtviz(signals, wavelet='db1', level=None, approx=None, cmap_name='seismic'):
    """
    params:
    -----
    signal:
        The signal or signals to be decomposed.

    wavelet:
        The wavelet to use. This can be either a string or a pywt.Wavelet object.

    levels:
        The number of levels to which the signal will be decomposed. Defaults to
        the maximum decomposition depth.

    approx:
        Boolean indicating whether the approximation coefficients will show up
        on the heatmap. Defaults to false if level is None, true otherwise.

    cmap_name:
        The name of the matplotlib colormap to use. Defaults to seismic. I
        recommend using a divergent colormap so that negative numbers are
        evident.

    returns:
    -----
    f:
        A matplotlib figure containing a heatmap of the wavelet coefficients and
        a plot of the signal.
    """

    # if we just have one signal, put it in a list
    if not isinstance(signals[0], collections.Iterable):
        signals = [signals]

    if approx is None:
        approx = level is not None

    f, ax = plt.subplots(((len(signals) + 1) // 2) * 2, min(2, len(signals)), squeeze=False, figsize=(15, 6))
    f.subplots_adjust(hspace=0.025)

    for i, signal in enumerate(signals):
        coefs = pywt.wavedec(signal, wavelet, level=level)
        if not approx:
            coefs = coefs[1:]
        max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)

        row = (i // 2) * 2
        col = i % 2

        heatmap_ax = ax[row][col]
        signal_ax = ax[row + 1][col]

        dwt_heatmap(coefs, heatmap_ax, cmap_name, approx, max_level, signal_ax)
        heatmap_ax.set_title('wavelet coefficient heatmap')
        signal_ax.plot(signal)
        signal_ax.set_xlim([0, len(signal)])
        signal_ax.set_title('signal', y = -0.20)
    return f

def dwt_heatmap(coefs, ax, cmap_name, approx, max_level, sig_ax):
    ax.set_xticks([])

    ax.set_yticks([(i / len(coefs)) - (1 / (len(coefs) * 2))
                   for i in range(len(coefs), 0, -1)])

    if not approx and len(coefs) == max_level:
        ax.set_yticklabels(range(1, max_level + 1))

    elif approx and len(coefs) == max_level + 1:
        ax.set_yticklabels(['approx'] + list(range(1, max_level + 1)))

    elif not approx and len(coefs) != max_level:
        ax.set_yticklabels(range(max_level - len(coefs) + 1, max_level + 1))

    elif approx and len(coefs) != max_level + 1:
        ax.set_yticklabels(['approx'] + list(range(max_level - len(coefs) + 2, max_level + 1)))
 
    ax.set_ylabel('levels')

    limit = max(abs(f(chain(*coefs))) for f in (max, min))
    norm = col.Normalize(vmin=-limit, vmax=limit)
    cmap = plt.get_cmap(cmap_name)

    colbar_axis = colbar.make_axes([ax, sig_ax], 'right')
    colbar.ColorbarBase(colbar_axis[0], cmap, norm)

    height = 1 / len(coefs)
    for level, coef_level in enumerate(coefs):
        width = 1 / len(coef_level)
        for n, coef in enumerate(coef_level):
            bottom_left = (0 + (n * width), 1 - ((level + 1) * height))
            color = cmap(norm(coef))
            heat_square = pat.Rectangle(bottom_left, width, height, color=color)
            ax.add_patch(heat_square)

