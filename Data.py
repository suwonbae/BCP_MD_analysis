import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Data2D:

    def __init__(self, z=None, name=None):

        self.z = z
        self.name = name

        self.plot_args = {}

    def plot(self, save=None, show=False, plot_args={}, **kwargs):

        self.plot_args.update(plot_args)
        print(plot_args)
        self._plot(save=save, show=show, **kwargs)

    def _plot(self, save=None, show=False, figsize=(4,3), **kwargs):

        self.fig, self.ax = plt.subplots( figsize=figsize )

        plot_args = self.plot_args.copy()
        plot_args.update(kwargs)

        zmin = plot_args['zmin']
        zmax = plot_args['zmax']
        cmap = plot_args['cmap']
        cbarticks = plot_args['cbarticks']
        cbarlabel = plot_args['cbarlabel']
        xticks = plot_args['xticks']
        xticklabels = plot_args['xticklabels']
        xlabel = plot_args['xlabel']
        yticks = plot_args['yticks']
        yticklabels = plot_args['yticklabels']
        ylabel = plot_args['ylabel']

        self.im = self.ax.imshow(self.z, vmin=zmin, vmax=zmax, cmap=cmap, origin='lower')
        self.ax.set_aspect('auto')
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = self.fig.colorbar(self.im, cax=cax)
        cbar.set_ticks(cbarticks)
        cbar.set_label(cbarlabel)
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xticklabels)
        self.ax.set_xlabel(xlabel)
        self.ax.set_yticks(yticks)
        self.ax.set_ylim(yticks[0] - 0.5, yticks[-1] + 0.5)
        self.ax.set_yticklabels(yticklabels)
        self.ax.set_ylabel(ylabel)
        plt.tight_layout(pad=1, h_pad=None, w_pad=None, rect=None)
        
        if save:
            plt.savefig(save, dpi=1000)
        
        if show:
            plt.show()
        
        plt.close(self.fig.number)

class Data_TimeSeries:

    def __init__(self, data, Nrepeat, std=False):

        ndim = data.ndim

        if ndim == 1:
            data = data.reshape(len(data), -1)
            binsize = 1
        if ndim == 2:
            binsize = data.shape[1]

        initial = data[0]
        rest = data[1:].reshape(-1, Nrepeat, binsize)

        mean = rest.mean(axis=1)
        self.mean = np.vstack((initial, mean))

        if std:
            std = rest.std(axis=1)
            self.std = np.vstack((np.zeros(binsize), std))
        
        if ndim == 1:
            self.mean = self.mean.reshape(-1)
            if std: self.std = self.std.reshape(-1)

