#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains basic helper functions for running a parameter sweep on the FAIR
cluster and make HiPlot server render your own experiments.
"""

from typing import Optional
from threading import Timer
import hiplot as hip
from hiplot.server import run_server
from parlai_diplomacy.utils.param_sweeps.collector import load_sweep

# Default hiplot server.
HIPLOT_SERVER_URL = "http://127.0.0.1:5005/"


def fetcher(uri) -> Optional[hip.Experiment]:
    """Prepare param sweep output for hiplot

    This function collect the param sweeping results and simplify them for easy display using hiplot

    :param uri: root dir that containing all the param_sweeping results.

    :returns: hiplot Experiment Object for display
    """
    df = load_sweep(uri, allow_multitask=False)
    if 'stdout' in df.columns:
        del df['stdout']
    if len(df) == 0:
        print("Errors parsing trainstats and opt files", uri)
        return None

    data = df.to_dict("records")
    exp = hip.Experiment.from_iterable(data)
    primary_metric = df.columns[-1]
    exp.display_data(hip.Displays.PARALLEL_PLOT).update(
        {"order": ["uid"] + list(df.columns)}
    )
    exp.parameters_definition[primary_metric].type = hip.ValueType.NUMERIC
    return exp


def open_browser():
    import webbrowser

    webbrowser.open(HIPLOT_SERVER_URL, new=2, autoraise=True)


def main():
    # By running the following command, a hiplot server will be rendered to display your experiment results
    # using the udf fetcher passed to hiplot.
    try:
        Timer(1, open_browser).start()
    except Exception as e:
        print("Fail to open browser", e)
    run_server(fetchers=[fetcher])


if __name__ == "__main__":
    main()
