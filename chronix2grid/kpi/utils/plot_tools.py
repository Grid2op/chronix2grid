# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import seaborn as sns


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
