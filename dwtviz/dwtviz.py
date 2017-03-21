from functools import partial
from multiprocessing import cpu_count, Pool
from itertools import chain
import sklearn.gaussian_process as sgp
import datetime
import numpy as np
import pywt
import matplotlib.ticker as mtk
import matplotlib.gridspec as grd
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.colors as col
import matplotlib.colorbar as colbar
from itertools import chain

def dwtviz(signals, wavelet='db1', level=None, approx=None, cmap_name='seismic',
           decomposition='dwt', cbar_limit=None, xyplot=False, index=True):
    """
    params:
    -----
    signal:
        The signal or signals to be decomposed.

    wavelet:
        The wavelet to use. This can be either a string or a pywt.Wavelet object.

    levels:
        The number of levels to which the signal will be decomposed. Defaults to
        the maximum decomposition depth
    .

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
    if type(signals) != list:
        signals = [signals]

    if approx is None:
        approx = level is not None

    if xyplot:
        nrows = len(signals)
        ncols = 2
    else:
        ncols = min(2, len(signals))
        nrows = (len(signals) + 1) // 2

    f = plt.figure(figsize=(10 * ncols, 7 * nrows))

    outer_gs = grd.GridSpec(nrows, ncols, hspace=.3, wspace=.1)

    all_coefs = []
    for signal in signals:
        if decomposition == 'dwt':
            coefs = pywt.wavedec(signal[1] if isinstance(signal, tuple) else signal, wavelet, level=level)
            if not approx:
                coefs = coefs[1:]
        elif decomposition == 'swt':
            coefs = pywt.swt(signal[1] if isinstance(signal, tuple) else signal, wavelet, level=level)
            coefs = [c[1] for c in coefs] 
        all_coefs.append(coefs)

    cbar_limit = cbar_limit if cbar_limit is not None else max(chain(*[np.abs(np.concatenate(c)) for c in all_coefs]))
    for i, signal in enumerate(signals):
        coefs = all_coefs[i]
        if xyplot:
            row = i
            col = 0
        else:
            row = i // 2
            col = i % 2

        gs = grd.GridSpecFromSubplotSpec(2, 1,
                       subplot_spec=outer_gs[row, col], hspace=0.2)
        
        max_level = pywt.dwt_max_level(len(signal[1] if isinstance(signal, tuple) else signal), pywt.Wavelet(wavelet).dec_len)

        if xyplot:
            gs2 = grd.GridSpecFromSubplotSpec(len(coefs), 1,
                                              subplot_spec=outer_gs[row, 1], hspace=0.1)

            y_axis = (np.min(coefs) - 1, np.max(coefs) + 1)
            for j, c in enumerate(coefs):
                ax = plt.subplot(gs2[j, 0])
                ax.set_ylim(y_axis)
                ax.plot(c)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_ylabel(j + 1, rotation=0)

        heatmap_ax = plt.subplot(gs[0, 0])
        signal_ax = plt.subplot(gs[1, 0])

        dwt_heatmap(coefs, heatmap_ax, cmap_name, approx, max_level, signal_ax, cbar_limit)
        if type(signal) == tuple:
            signal_ax.plot(*signal)
        else:
            signal_ax.plot(signal)

        signal_ax.set_xlim([min(signal[0]), max(signal[0])] if type(signal) == tuple else [0, len(signal) - 1])
        signal_ax.set_xticks([])

        if index:
            heatmap_ax.set_title(i)
    return f

def dwt_heatmap(coefs, ax, cmap_name, approx, max_level, sig_ax, cbar_limit):
    ax.set_xticks(np.array(list(range(0, len(coefs[0]), 5))) / len(coefs[0]))
    ax.set_xticklabels(range(0, len(coefs[-1]), 5))

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

    norm = col.Normalize(vmin=-cbar_limit, vmax=cbar_limit)
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

def dwtviz_gp(signals, length=None, samples=8, kernel=None, xseconds=False,
              decomposition='dwt', cbar_limit=None, xyplot=False, truncate=False,
              noise_tolerance=5):
    """eu
    signals: a list of tuples, where the first element is X values and the second is Y values.
    """
    gp_signals, truncated_signals = fit_gps(signals, length, samples, kernel, truncate)
    
    fig = dwtviz(list(gp_signals), decomposition=decomposition, cbar_limit=cbar_limit, xyplot=xyplot)
    
    fig = add_original_scatter(truncated_signals, fig, xseconds, xyplot=xyplot)
    return fig

def seconds_converter(seconds, _):
    return str(datetime.timedelta(seconds=seconds))
seconds_formater = mtk.FuncFormatter(seconds_converter)

def fit_gps(signals, length=None, samples=8, kernel=None, noise_tolerance=5, truncate=False):
    if kernel is None:
        kernel = ( 
                sgp.kernels.ConstantKernel(1) * sgp.kernels.RBF(1e7, (1e6, 1e8))        
                + sgp.kernels.ConstantKernel(1) * sgp.kernels.RBF(1.5e6, (1e6, 1e7))       
                + sgp.kernels.ConstantKernel(.0001) * sgp.kernels.RBF(1.5e4, (1e4, 1e5))
        )
    gp = sgp.GaussianProcessRegressor(kernel=kernel, alpha=noise_tolerance, n_restarts_optimizer=12)

    if length is None:
        longest = max(max(x) for x, y in signals)
    else:
        longest = length

    fit_gp_to_length = partial(fit_gp, longest, gp, 2**samples, truncate)
    with Pool(cpu_count() - 1) as p:
        results = p.map(fit_gp_to_length, signals)
    return zip(*results)

def fit_gp(longest, gp, num_samples, truncate, signal):
    x, y = signal
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    gp.fit(x, y)
    xnew = np.linspace(0, longest, num_samples).reshape(-1, 1)
    ynew = gp.predict(xnew).flatten()

    length = max(x) - min(x)

    if length is None:
        prop = length[0] / longest
    else:
        prop = 1

    if truncate:
        i = int(np.log2(1/prop)) + 1
    else:
        i = 0

    end = num_samples // (2**i)
    gp_signal = (xnew[:end].flatten(), ynew[:end])
    truncated_xs = [a for a in x.flatten() if a <= longest // (2 ** i)]
    truncated_signal = (truncated_xs, y[:len(truncated_xs)].flatten())
    return (gp_signal, truncated_signal)

def add_original_scatter(signals, dwtviz_fig, xseconds=True, xyplot=False):
    for i, s in enumerate(signals):
        if xyplot:
            # TODO: this will only work for 6-level decompositions
            # generalize to arbitrarily deep decompositions
            plot_index = 7 + (i * 9)  
        else:
            plot_index = 1 + (i * 3)
        ax = dwtviz_fig.axes[plot_index]
        if xseconds:
            ax.xaxis.set_major_formatter(seconds_formater)
        xs = s[0]
        ax.set_xticks(np.linspace(xs[0], xs[-1], 4))
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
        ax.scatter(*s)
    return dwtviz_fig 
