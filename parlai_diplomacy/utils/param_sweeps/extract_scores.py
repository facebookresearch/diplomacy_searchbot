#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Functions for extracting relevant information from a directory containing the
results of a parameter sweep for models trained using the ParlAI framework.
"""

import ast
import math
import os


def extract_scores(text, metric, other=None, print_metrics=True):
    """Helper function to extract scores from the stdout."""
    run_dict = {}
    primary_list = []
    if len(metric.split(':')) == 4:
        primary_metric = metric.split(':')[1] + ':' + metric.split(':')[2]
        primary_set = metric.split(':')[3]
    else:
        primary_metric = metric.split(':')[1]
        if len(metric.split(':')) > 2:
            primary_set = metric.split(':')[2]
        else:
            primary_set = 'train'
    if '/' in primary_metric:
        subtask = primary_metric.split('/')[0]
        primary_metric_sub = primary_metric.split('/')[1]
    primary_optim = metric.split(':')[0]
    if other is not None:
        other_metrics = []
        other_optims = []
        other_lists = [[] for _ in range(len(other))]
        other_sets = []
        for metric in other:
            other_metrics.append(metric.split(':')[1])
            other_optims.append(metric.split(':')[0])
            if len(metric.split(':')) > 2:
                other_set = metric.split(':')[2]
            else:
                other_set = 'train'
            other_sets.append(other_set)
    total = 0
    list_output = text.split('\n')
    for line in list_output:
        if '[ time:' in line or 'valid:' in line or 'test:' in line:
            if 'valid:' in line:
                try:
                    if 'stream' in line:
                        score_dict = ast.literal_eval(line.split(':', 2)[2])
                    else:
                        score_dict = ast.literal_eval(line.split(':', 1)[1])
                    if '/' in primary_metric:
                        if (
                            subtask in score_dict['tasks']
                            and primary_metric_sub in score_dict['tasks'][subtask]
                            and primary_set == 'valid'
                        ):
                            primary_list.append(
                                float(score_dict['tasks'][subtask][primary_metric_sub])
                            )
                    elif primary_metric in score_dict and primary_set == 'valid':
                        primary_list.append(float(score_dict[primary_metric]))
                    if other is not None:
                        for i in range(len(other_metrics)):
                            if (
                                other_metrics[i] in score_dict
                                and other_sets[i] == 'valid'
                            ):
                                other_lists[i].append(score_dict[other_metrics[i]])
                except:
                    pass
            elif 'test:' in line:
                try:
                    if 'stream' in line:
                        score_dict = ast.literal_eval(line.split(':', 2)[2])
                    else:
                        score_dict = ast.literal_eval(line.split(':', 1)[1])
                    if '/' in primary_metric:
                        if (
                            subtask in score_dict['tasks']
                            and primary_metric_sub in score_dict['tasks'][subtask]
                            and primary_set == 'test'
                        ):
                            primary_list.append(
                                float(score_dict['tasks'][subtask][primary_metric_sub])
                            )
                    elif primary_metric in score_dict and primary_set == 'test':
                        primary_list.append(float(score_dict[primary_metric]))
                    if other is not None:
                        for i in range(len(other_metrics)):
                            if (
                                other_metrics[i] in score_dict
                                and other_sets[i] == 'test'
                            ):
                                other_lists[i].append(score_dict[other_metrics[i]])
                except:
                    pass
            else:
                try:
                    score_dict = ast.literal_eval(line.split('] ', 1)[1])
                    total += int(score_dict['exs'])
                    if '/' in primary_metric:
                        if (
                            subtask in score_dict['tasks']
                            and primary_metric_sub in score_dict['tasks'][subtask]
                            and primary_set == 'train'
                        ):
                            primary_list.append(
                                float(score_dict['tasks'][subtask][primary_metric_sub])
                            )
                    elif primary_metric in score_dict and primary_set == 'train':
                        primary_list.append(float(score_dict[primary_metric]))
                    if other is not None:
                        for i in range(len(other_metrics)):
                            if (
                                other_metrics[i] in score_dict
                                and other_sets[i] == 'train'
                            ):
                                other_lists[i].append(score_dict[other_metrics[i]])
                except:
                    pass

    if print_metrics:
        print(primary_metric, '(' + primary_set + '):', primary_list, '\n~')
    if primary_list:
        if primary_optim == 'min':
            if print_metrics:
                print(
                    'Min',
                    primary_metric,
                    '(' + primary_set + '):',
                    min(primary_list),
                    '\n~',
                )
            run_dict['primary_optim'] = min(primary_list)
        elif primary_optim == 'max':
            if print_metrics:
                print(
                    'Max',
                    primary_metric,
                    '(' + primary_set + '):',
                    max(primary_list),
                    '\n~',
                )
            run_dict['primary_optim'] = max(primary_list)
    else:
        run_dict['primary_optim'] = None
    if other is not None:
        for i in range(len(other_metrics)):
            if other_lists[i]:
                if print_metrics:
                    print(
                        other_metrics[i],
                        '(' + other_sets[i] + '):',
                        other_lists[i],
                        '\n~',
                    )
                if other_optims[i] == 'min':
                    if print_metrics:
                        print(
                            'Min',
                            other_metrics[i],
                            '(' + other_sets[i] + '):',
                            min(other_lists[i]),
                            '\n~',
                        )
                    run_dict[
                        'best-' + str(other_metrics[i]) + '(' + str(other_sets[i]) + ')'
                    ] = min(other_lists[i])
                elif other_optims[i] == 'max':
                    if print_metrics:
                        print(
                            'Max',
                            other_metrics[i],
                            '(' + other_sets[i] + '):',
                            max(other_lists[i]),
                            '\n~',
                        )
                    run_dict[
                        'best-' + str(other_metrics[i]) + '(' + str(other_sets[i]) + ')'
                    ] = max(other_lists[i])
                run_dict[other_metrics[i] + ' (' + other_sets[i] + ')'] = other_lists[i]
            else:
                run_dict[
                    'best-' + str(other_metrics[i]) + '(' + str(other_sets[i]) + ')'
                ] = None

    run_dict[primary_metric] = primary_list

    return run_dict


if __name__ == '__main__':
    """
    This prints out the metrics you specify for your run, and collects the best
    model from your specified metrics.

    dir - specify the root directory containing your sweep (ex. /checkpoint/edinan/20180213/twitter_param_sweep)
    metric - this is the metric we optimize for (ex. max:accuracy:valid)
    other-metrics - list other metrics you want to print out in same format as above
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Crawl through sweep to find best model'
    )
    parser.add_argument('--dir', type=str, help='directory containing sweep results')
    parser.add_argument(
        '--metric',
        type=str,
        help='will search for model optimal value of this metric; specify how to optimize with ":", e.g. "min:loss".'
        'If only want metric for test or valid, specify with min:loss:valid or min:loss:test',
    )
    parser.add_argument(
        '--other-metrics', nargs='*', type=str, help='other metrics to report on'
    )
    parser.add_argument(
        '--hide-err', action='store_true', default=False, help='hide stderr'
    )
    parser.add_argument('--sorted', action='store_true', default=False)
    args = parser.parse_args()

    if not args.dir:
        print('Must specify root directory with --dir. Quitting')
        quit()
    if not args.metric:
        print(
            'Must specify metric for which to search for optimal value (e.g., accuracy, loss) with --metric. Quitting'
        )
        quit()

    root = args.dir
    print_err = not args.hide_err
    dicts = []

    primary_metric = args.metric.split(':')[1]
    primary_optim = args.metric.split(':')[0]
    if len(args.metric.split(':')) > 2:
        primary_set = args.metric.split(':')[2]
    else:
        primary_set = 'train'

    if primary_optim == 'min':
        primary_value = math.inf
    else:
        primary_value = -math.inf
    best_model = 0

    scores_and_names = []

    i = 0
    for subdir, dirs, files in os.walk(root, followlinks=True):
        if 'slurm_logs' in subdir:
            continue
        for filename in files:
            if 'stdout' in str(filename):
                filepath = subdir + os.sep + filename
                print('==============================================')
                print('File: ', filepath)
                print('Report: ', '\n~')
                text = open(filepath, 'r').read()
                run_dict = extract_scores(text, args.metric, args.other_metrics)
                run_dict['filepath'] = filepath
                dicts.append(run_dict)
                if (
                    'primary_optim' in run_dict
                    and run_dict['primary_optim'] is not None
                ):
                    optims = [str(run_dict['primary_optim'])]
                    for k, v in run_dict.items():
                        if 'best' in k:
                            optims.append(str(v))
                    scores_and_names.append((','.join(optims), filepath))
                    if primary_optim == 'min':
                        if run_dict['primary_optim'] < primary_value:
                            primary_value = run_dict['primary_optim']
                            best_model = i
                    else:
                        if run_dict['primary_optim'] > primary_value:
                            primary_value = run_dict['primary_optim']
                            best_model = i
                i += 1

            # Prints out error log if it is not empty
            if print_err and 'stderr' in str(filename):
                filepath = subdir + os.sep + filename
                if os.stat(filepath).st_size != 0:
                    print('ERROR LOG:')
                    try:
                        text = open(filepath, 'r').read()
                        print(text)
                    except:
                        print("could not print error log")

    # Now print information about the best model based on the specified metrics
    if dicts:
        print('==============================================')
        print('==============================================')
        print('BEST MODEL: ', dicts[best_model]['filepath'])
        for k, v in dicts[best_model].items():
            if k != 'filepath' and 'best' not in k:
                if k == 'primary_optim':
                    print(primary_metric, '(' + primary_set + '):', v)
                    print('~')
                else:
                    print(k, ':', v)
                    print('~')
        if args.sorted:

            def cmp(x):
                split = x[0].split(',')
                ret = []
                for s in split:
                    try:
                        ret.append(float(s))
                    except TypeError:
                        ret.append(s)
                return ret

            scores_and_names.sort(reverse=primary_optim == 'min', key=cmp)
            print('\n'.join(['{}: {}'.format(e[0], e[1]) for e in scores_and_names]))
    else:
        print('No information yet')
