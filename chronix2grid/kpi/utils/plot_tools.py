import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product
from pathlib import Path
from matplotlib import pyplot as plt

def _plot_heatmap(self, corr, title, path_png=None, save_png=False):

    ax = sns.heatmap(corr,
                     fmt='.1f',
                     annot = False,
                     vmin=-1, vmax=1, center=0,
                     cmap=sns.diverging_palette(20, 220, n=200),
                     cbar_kws={"orientation": "horizontal",
                               "shrink": 0.35,
                               },
                     square=True,
                     linewidths = 0.5,
                     )
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=45,
                       horizontalalignment='right'
                       )

    ax.set_title(title, fontsize=15)

    if save_png:
        figure = ax.get_figure()
        figure.savefig(path_png)
