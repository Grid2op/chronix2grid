# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product
from pathlib import Path
from matplotlib import pyplot as plt

class NuclearKPI:

    def __init__(self, kpi_databanks):
        self.kpi_databanks = kpi_databanks

    def nuclear_kpi(self, save_plots=False):

        """
        Return:
        ------

        None
        """

        # Get the nuclear gen names
        nuclear_filter = self.prod_charac.type.isin(['nuclear'])
        nuclear_names = self.prod_charac.name.loc[nuclear_filter].values

        # Extract only nuclear power plants
        nuclear_ref = self.ref_dispatch[nuclear_names]
        nuclear_syn = self.syn_dispatch[nuclear_names]

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(nuclear_ref.values, bins=100)
        axes[1].hist(nuclear_syn.values, bins=100)
        axes[0].set_title('Nuclear Reference Distribution')
        axes[1].set_title('Nuclear Synthetic Distribution')

        if save_plots:
            # Save plot as pnd
            extent0 = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            extent1 = axes[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig('images/nuclear_kpi/ref_dispatch_histogram.png', bbox_inches=extent0.expanded(1.3, 1.3))
            fig.savefig('images/nuclear_kpi/syn_dispatch_histogram.png', bbox_inches=extent1.expanded(1.3, 1.3))

        return None
