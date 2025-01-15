import datetime
from functools import partial
from multiprocessing import cpu_count, Pool

import matplotlib.ticker as mtk

import sklearn.gaussian_process as sgp


def seconds_converter(seconds, _):
    return str(datetime.timedelta(seconds=seconds))


seconds_formater = mtk.FuncFormatter(seconds_converter)


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


def dwtviz_gp(
    signals,
    length=None,
    samples=8,
    kernel=None,
    xseconds=False,
    decomposition="dwt",
    cbar_limit=None,
    xyplot=False,
    truncate=False,
    noise_tolerance=5,
):
    """eu
    signals: a list of tuples, where the first element is X values and the
    second is Y values.
    """
    gp_signals, truncated_signals = fit_gps(signals, length, samples, kernel, truncate)

    fig = dwtviz(
        list(gp_signals),
        decomposition=decomposition,
        cbar_limit=cbar_limit,
        xyplot=xyplot,
    )

    fig = add_original_scatter(truncated_signals, fig, xseconds, xyplot=xyplot)
    return fig


def fit_gps(
    signals, length=None, samples=8, kernel=None, noise_tolerance=5, truncate=False
):
    if kernel is None:
        kernel = (
            sgp.kernels.ConstantKernel(1) * sgp.kernels.RBF(1e7, (1e6, 1e8))
            + sgp.kernels.ConstantKernel(1) * sgp.kernels.RBF(1.5e6, (1e6, 1e7))
            + sgp.kernels.ConstantKernel(0.0001) * sgp.kernels.RBF(1.5e4, (1e4, 1e5))
        )
    gp = sgp.GaussianProcessRegressor(
        kernel=kernel, alpha=noise_tolerance, n_restarts_optimizer=12
    )

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
        i = int(np.log2(1 / prop)) + 1
    else:
        i = 0

    end = num_samples // (2**i)
    gp_signal = (xnew[:end].flatten(), ynew[:end])
    truncated_xs = [a for a in x.flatten() if a <= longest // (2**i)]
    truncated_signal = (truncated_xs, y[: len(truncated_xs)].flatten())
    return (gp_signal, truncated_signal)
