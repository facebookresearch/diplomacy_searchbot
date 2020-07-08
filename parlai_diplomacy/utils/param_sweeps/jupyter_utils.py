#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Functions for extracting relevant information from a directory containing the
results of a parameter sweep for models trained using the ParlAI framework.

These functions are for use within jupyter to analyze and plot data.
"""

from .extract_scores import extract_scores

import math
import os

import numpy as np
import pandas as pd


def jupyter_summary(root, metric, other_metrics=None, print_summary=False):
    """Call this from a jupyter notebook to collect a summary of your sweep.
    root - root directory containing your sweep
    metric - metric we want to optimize for when collecting the 'best' model (ex. min:loss:test)
    other_metrics - list containing other metrics we want to report on (ex. [max:accuracy:valid, min:loss:train])
    print_summary - print a summary containing information about metrics from all models in sweep
    Returns a pandas DataFrame.
    """
    summary_dict = {}
    summary_dict_metrics = {}
    dicts = []
    primary_optim = metric.split(':')[0]
    if len(metric.split(':')) == 4:
        primary_metric = metric.split(':')[1] + ':' + metric.split(':')[2]
        primary_set = metric.split(':')[3]
    else:
        primary_metric = metric.split(':')[1]
        if len(metric.split(':')) > 2:
            primary_set = metric.split(':')[2]
        else:
            primary_set = 'train'

    if primary_optim == 'min':
        primary_value = math.inf
    else:
        primary_value = -math.inf
    best_model = 0

    # filepath will be added to every row
    summary_dict['filepath'] = []

    i = 0
    for subdir, dirs, files in os.walk(root):
        for filename in files:
            if 'stdout' in str(filename):
                str_subdir = str(subdir).split('/')[-1]
                hyperparameters = str_subdir.split('_')
                for param in hyperparameters:
                    param_values = param.split('=')
                    summary_dict.setdefault(param_values[0], []).append(param_values[1])
                filepath = subdir + os.sep + filename
                text = open(filepath, 'r').read()
                if print_summary:
                    print('==============================================')
                    print('File: ', filepath)
                    print('Report: ', '\n~')
                run_dict = extract_scores(
                    text, metric, other_metrics, print_metrics=print_summary
                )
                run_dict['filepath'] = filepath
                dicts.append(run_dict)
                summary_dict['filepath'].append(subdir)
                if 'primary_optim' in run_dict:
                    if run_dict['primary_optim'] is not None:
                        if primary_optim == 'min':
                            if run_dict['primary_optim'] < primary_value:
                                primary_value = run_dict['primary_optim']
                                best_model = i
                        else:
                            if run_dict['primary_optim'] > primary_value:
                                primary_value = run_dict['primary_optim']
                                best_model = i
                    key = str(primary_metric) + '(' + str(primary_set) + ')'
                    summary_dict_metrics.setdefault(key, []).append(
                        run_dict['primary_optim']
                    )
                for k, v in run_dict.items():
                    if 'best' in k:
                        key = k.split('-', 1)[1]
                        summary_dict_metrics.setdefault(key, []).append(run_dict[k])
                i += 1
            if print_summary:
                if 'stderr' in str(filename):
                    filepath = subdir + os.sep + filename
                    if os.stat(filepath).st_size != 0:
                        print('ERROR LOG:')
                        text = open(filepath, 'r').read()
                        print(text)

    if print_summary:
        print('==============================================')
        print('~')
        print('BEST MODEL: ', dicts[best_model]['filepath'])
        for k, v in dicts[best_model].items():
            if k != 'filepath':
                if k == 'primary_optim':
                    print(primary_metric, '(' + primary_set + '):', v)
                    print('~')
                else:
                    print(k, ':', v)
                    print('~')

    for k, v in summary_dict_metrics.items():
        summary_dict[k.upper()] = v
    df = pd.DataFrame(summary_dict)
    # sort by primary metric
    k = str(primary_metric) + '(' + str(primary_set) + ')'
    df = df.sort_values(k.upper())
    return df


def jupyter_collect_best(root, metric, other_metrics=None):
    """Call this from a jupyter notebook to collect info about the best model
    contained in your root directory based on your specified metric.

    root - root directory containing your sweep
    metric - metric we want to optimize for when collecting the 'best' model (ex. min:loss:test)
    other_metrics - list containing other metrics we want to report on (ex. [max:accuracy:valid, min:loss:train])

    Returns a list of pandas DataFrames.
    """
    dicts = []
    primary_optim = metric.split(':')[0]
    if len(metric.split(':')) == 4:
        primary_metric = metric.split(':')[1] + ':' + metric.split(':')[2]
        primary_set = metric.split(':')[3]
    else:
        primary_metric = metric.split(':')[1]
        if len(metric.split(':')) > 2:
            primary_set = metric.split(':')[2]
        else:
            primary_set = 'train'

    if primary_optim == 'min':
        primary_value = math.inf
    else:
        primary_value = -math.inf
    best_model = 0

    i = 0
    for subdir, dirs, files in os.walk(root):
        for filename in files:
            if 'stdout' in str(filename):
                filepath = subdir + os.sep + filename
                text = open(filepath, 'r').read()
                run_dict = extract_scores(
                    text, metric, other_metrics, print_metrics=False
                )
                run_dict['filepath'] = filepath
                dicts.append(run_dict)
                if (
                    'primary_optim' in run_dict
                    and run_dict['primary_optim'] is not None
                ):
                    if primary_optim == 'min':
                        if run_dict['primary_optim'] < primary_value:
                            primary_value = run_dict['primary_optim']
                            best_model = i
                    else:
                        if run_dict['primary_optim'] > primary_value:
                            primary_value = run_dict['primary_optim']
                            best_model = i
                i += 1

    # Create dataframes from data collected for best model
    # Three possible dataframes (train, valid, test)
    datas = [
        {primary_metric + ' (' + primary_set + ')': dicts[best_model][primary_metric]},
        None,
        None,
    ]
    lengths = [len(dicts[best_model][primary_metric]), 0, 0]

    for k, v in dicts[best_model].items():
        if (
            k != primary_metric
            and k != 'filepath'
            and k != 'primary_optim'
            and 'best' not in k
        ):
            found = False
            for i in range(len(lengths)):
                if not found:
                    if len(v) == lengths[i] and lengths[i] > 0:
                        datas[i][k] = v
                        found = True
                    elif lengths[i] == 0:
                        datas[i] = {k: v}
                        lengths[i] = len(v)
                        found = True

    data_frames_list = [pd.DataFrame(data) for data in datas if data is not None]

    return data_frames_list


def visualize_df(df_list):
    """
    This function takes a list of dataframes and creates plots
    """
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    # plot style
    fontP = FontProperties(family='serif', size=17)
    plt.style.use('ggplot')
    font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 18}

    for df in df_list:
        for k, v in df.items():
            plt.figure()
            plt.title(k, fontdict=font)
            plt.plot(v, color='darkseagreen', linewidth=4)

    plt.show()
