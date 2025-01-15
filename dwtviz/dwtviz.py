from itertools import chain
import numpy as np
import pywt
import matplotlib.gridspec as grd
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.colors as col
import matplotlib.colorbar as colbar


def dwtviz(
    signals,
    wavelet="db1",
    level=None,
    approx=None,
    cmap_name="seismic",
    decomposition="dwt",
    cbar_limit=None,
    xyplot=False,
    xticks=True,
    yticks=True,
    index=True,
):
    """
    params:
    -----
    signal:
        The signal or signals to be decomposed.

    wavelet:
        The wavelet to use. This can be either a string or a pywt.Wavelet
        object.

    levels:
        The number of levels to which the signal will be decomposed. Defaults
        to the maximum decomposition depth
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
        A matplotlib figure containing a heatmap of the wavelet coefficients
        and a plot of the signal.
    """

    if not isinstance(signals, list):
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

    outer_gs = grd.GridSpec(nrows, ncols, hspace=0.3, wspace=0.1)

    all_coefs = []
    for signal in signals:
        if decomposition == "dwt":
            coefs = pywt.wavedec(
                signal[1] if isinstance(signal, tuple) else signal, wavelet, level=level
            )
            if not approx:
                coefs = coefs[1:]
        elif decomposition == "swt" or decomposition == "sdwt":
            swt_decomp = pywt.swt(
                signal[1] if isinstance(signal, tuple) else signal, wavelet, level=level
            )
            coefs = [c[1] for c in swt_decomp]
            if approx:
                coefs = [swt_decomp[0][0]] + coefs

            if decomposition == "sdwt":
                smallest = -np.max(np.abs(np.array(coefs)))
                abbreviated = []
                if approx:
                    ap = np.ones_like(coefs[0]) * smallest
                    part = coefs[0][: -(2**level - 1)]
                    ap[: len(part)] = part
                    abbreviated.append(ap)
                    coefs = coefs[1:]

                for l, c in enumerate(coefs):
                    ap = np.ones_like(coefs[l]) * smallest
                    part = coefs[l][: -(2 ** (level - l) - 1)]
                    ap[: len(part)] = part
                    abbreviated.append(ap)

                coefs = abbreviated

        all_coefs.append(coefs)

    cbar_limit = (
        cbar_limit
        if cbar_limit is not None
        else max(chain(*[np.abs(np.concatenate(c)) for c in all_coefs]))
    )
    for i, signal in enumerate(signals):
        coefs = all_coefs[i]
        if xyplot:
            row = i
            col = 0
        else:
            row = i // 2
            col = i % 2

        gs = grd.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[row, col], hspace=0.2
        )

        max_level = pywt.dwt_max_level(
            len(signal[1] if isinstance(signal, tuple) else signal),
            pywt.Wavelet(wavelet).dec_len,
        )

        if xyplot:
            gs2 = grd.GridSpecFromSubplotSpec(
                len(coefs), 1, subplot_spec=outer_gs[row, 1], hspace=0.1
            )

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

        dwt_heatmap(
            coefs,
            heatmap_ax,
            cmap_name,
            approx,
            max_level,
            signal_ax,
            cbar_limit,
            xticks,
            yticks,
        )
        if isinstance(signal, tuple):
            signal_ax.plot(*signal)
        else:
            signal_ax.plot(signal)

        signal_ax.set_xlim(
            [min(signal[0]), max(signal[0])]
            if isinstance(signal, tuple)
            else [0, len(signal) - 1]
        )
        signal_ax.set_xticks([])

        if index:
            heatmap_ax.set_title(i)
    return f


def dwt_heatmap(
    coefs, ax, cmap_name, approx, max_level, sig_ax, cbar_limit, xticks, yticks
):
    if xticks:
        ax.set_xticks(np.array(list(range(0, len(coefs[0]), 5))) / len(coefs[0]))
        ax.set_xticklabels(range(0, len(coefs[-1]), 5))
    else:
        ax.set_xticks([])

    if yticks:
        ax.set_yticks(
            [
                (i / len(coefs)) - (1 / (len(coefs) * 2))
                for i in range(len(coefs), 0, -1)
            ]
        )

        if not approx and len(coefs) == max_level:
            ax.set_yticklabels(reversed(range(1, max_level + 1)))

        elif approx and len(coefs) == max_level + 1:
            labels = ["approx"] + list(reversed(range(1, max_level + 1)))
            ax.set_yticklabels(labels)

        elif not approx and len(coefs) != max_level:
            ax.set_yticklabels(range(max_level - len(coefs) + 1, max_level + 1))

        elif approx and len(coefs) != max_level + 1:
            ax.set_yticklabels(["approx"] + list(reversed(range(1, len(coefs)))))

    ax.set_ylabel("levels")

    norm = col.Normalize(vmin=-cbar_limit, vmax=cbar_limit)
    cmap = plt.get_cmap(cmap_name)

    colbar_axis = colbar.make_axes([ax, sig_ax], "right")
    colbar.ColorbarBase(colbar_axis[0], cmap, norm)

    height = 1 / len(coefs)
    for level, coef_level in enumerate(coefs):
        width = 1 / len(coef_level)
        for n, coef in enumerate(coef_level):
            bottom_left = (0 + (n * width), 1 - ((level + 1) * height))
            color = cmap(norm(coef))
            heat_square = pat.Rectangle(bottom_left, width, height, color=color)
            ax.add_patch(heat_square)
