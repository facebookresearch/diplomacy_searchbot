#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import sys
import glob
import argparse
import collections
import json
import getpass
import pandas as pd
import shutil
from parlai.utils.misc import float_formatter
from scipy.spatial.distance import cosine

# Axes to skipped for visualizing hyperparameter tuning
SKIP_KEYS = [
    "exs",
    "examples",
    "tokens_per_batch",
    "tbp",
    "gpu_mem_percent",
    "gpu_mem",
    "starttime",
    "lr",
    "model_file",
    "loss",
    "persona_path",
    "batchindex",
    "dict_file",
    "parlai_home",
    "model_name",
    "model",
]

# Directories to skip while collecting param sweeping results
SKIP_DIRS = ["ParlAI"]

TRAINSTATS_FILENAME = "model.checkpoint.trainstats"
TRAINSTATS_FILENAME2 = "model.trainstats"
OPT_FILE_NAME = "model.opt"
MODEL_FILENAME = "model"
CHECKPOINT_FILENAME = "model.checkpoint"

# multitask names longer than this will be cut off
MAX_FILENAME_WIDTH = 30
MAX_COLUMN_WIDTH = 30


def _flatten_nested_list(d, flatten_multi_item=False, parent_key="", sep="___") -> dict:
    """
    Flatten list-type dictionary values.

    This function serve 2 ways of standarding dictionaries containing list-type
    values:

        1. casting list to strings by default (flatten_multi_item = False)
        2. iterating the dict by key/value, creats new keys for your new
           dictionary and creating

    The dictionary at final step. It also try to standardize dictionary
    containing one-element list and convert to scalar.

    :param d:
        the dictionary to be flattened/normalized.
    :param flatten_multi_item:
        if set True, the multi-element list will be flatten to multiple (key,
        value) pairs to extended in d. For example:

        ``{'d': [0.1,0.2]} to {'d___0': 0.1, 'd___1':0.2}``.

    :param parent_key:
        key value to concat for the 0-layer
    :param sep:
        delimiter for creating new keys iteratively.
    :type d:
        dict
    :type flatten_multi_item:
        bool
    :type parent_key:
        tr
    :type sep:
        str

    :returns:
        dataframe of training output
    """

    for x in d:
        if type(d[x]) == list:
            if len(d[x]) == 1:
                d[x] = d[x][0]
            elif not flatten_multi_item:
                d[x] = "[" + ",".join(str(e) for e in d[x]) + "]"
    if not flatten_multi_item:
        return d
    res = []
    if not isinstance(d, collections.MutableMapping):
        res.append((parent_key, d))
        return dict(res)
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            res.extend(_flatten_nested_list(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for idx, value in enumerate(v):
                res.extend(
                    _flatten_nested_list(value, new_key + sep + str(idx), sep).items()
                )
        else:
            res.append((new_key, v))
    return dict(res)


def _find_sweep_folders(root: str):
    for dn, dirs, files in os.walk(root):
        # Exclude directories in os.walk
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        if (TRAINSTATS_FILENAME not in files and TRAINSTATS_FILENAME2 not in files) or (
            OPT_FILE_NAME not in files
        ):
            continue

        yield dn


def _flatten(item):
    if isinstance(item, (list, tuple)):
        return str(item)
    else:
        return item


def _short_teacher(teachername):
    if teachername.startswith('/'):
        # keep only the file two paths of a filename
        return os.path.join(
            os.path.basename(os.path.dirname(teachername)),
            os.path.basename(teachername),
        )[-MAX_FILENAME_WIDTH:]
    if ':' in teachername:
        teachername = (
            teachername.replace("parlai.tasks.", "")
            .replace("parlai_internal.tasks.", "")
            .replace("parlai_diplomacy.tasks.", "")
            .replace('.agents', '')
            .replace('Teacher', '')
        )
    return teachername


def _split_key(keyname):
    fields = keyname.split("/")
    key = fields.pop(-1)
    if fields:
        teacher = "/".join(fields)
        return (_short_teacher(teacher), key)
    else:
        return ('', key)


def load_sweep(
    args,
    root: str,
    best_only: bool = True,
    filter_columns: bool = True,
    sort_metric=None,
    sort_mode=None,
    allow_multitask=True,
    include_stdout=True,
    max_rows=0,
    metrics=None,
    filter_opt=None,
    ignore_fields=None,
) -> pd.DataFrame:
    """
    Grep results from training logs.

    This util function parse various evaluation metrics for validation sets
    from .trainstats files and model hyperparameters settings from .opt files
    under the directory root and merge them two.

    :param root:
        root path containing all the raw output
    :param best_only:
        If true (default), reports only the best results from each item in sweep.
        You will have exactly one row per sweep run.
    :param filter_columns:
        if true (default), unchanged hyperparameters will be filtered. if false,
        all columns will be completely unfiltered
    :param allow_multitask:
        if True, the resulting dataframe may have MultiIndex columns, indicating
        grouped metrics by multitasking. If false, all multitasking submetrics
        are hidden.
    :param max_rows:
        if a positive integer, returns only the N best rows.
    :param filter_opt:
        delimited list of opt key/value pairs to filter by
    :param ignore_fields:
        list of fields to not visualize

    :returns:
        dataframe that merges evaluation metrics for validation set and model
        settings
    """
    # final results to be returned
    results = []

    # primary validation metrics for model performance, parsed from .opt file
    primary_metric = sort_metric
    sort_direction = sort_mode

    nodelete_keys = {'valno', 'stdout', 'uid'}
    metrics_keys = set()

    for sweep_folder in _find_sweep_folders(root):
        # Load .opt file containing model hyperparameters
        with open(os.path.join(sweep_folder, OPT_FILE_NAME), "r") as f:
            opt = json.load(f)
        if "override" in opt:
            ov = opt["override"]
            del opt["override"]
            opt = {**opt, **ov}
        opt = {k: _flatten(v) for k, v in opt.items()}

        # if filter opt is not None, check if this sweep path matches criteria
        if filter_opt is not None and not all(
            str(opt.get(k)) == v for k, v in filter_opt
        ):
            continue

        if 'validation_metric' in opt and primary_metric is None:
            primary_metric = opt['validation_metric']
            sort_direction = opt['validation_metric_mode']

        # Load .trainstats file summarizing model performance
        try:
            json_filename = os.path.join(sweep_folder, TRAINSTATS_FILENAME)
            with open(json_filename, "r") as json_file:
                metrics_dict = json.load(json_file)
                valid_reports = metrics_dict['valid_reports']
        except FileNotFoundError:
            json_filename = os.path.join(sweep_folder, TRAINSTATS_FILENAME2)
            with open(json_filename, "r") as json_file:
                metrics_dict = json.load(json_file)
                valid_reports = metrics_dict['valid_reports']

        # add the last stdout file as the main one
        stdout = sorted(glob.glob(os.path.join(sweep_folder, "stdout.*")))[-1]
        stdout = stdout.replace(root, '')
        if stdout.startswith('/'):
            stdout = stdout[1:]

        for i, valid_report in enumerate(valid_reports):
            # clean up some old stuff
            for k in ['warning', 'examples']:
                if k in valid_report:
                    del valid_report[k]
            if 'tasks' in valid_report:
                tasks = valid_report.pop('tasks')
                for task, submetrics in tasks.items():
                    for submetric, value in submetrics.items():
                        valid_report[f'{task}/{submetric}'] = value
            if args.max_train_updates > -1:
                if (
                    'total_train_updates' in valid_report
                    and valid_report['total_train_updates'] > args.max_train_updates
                ):
                    for k in copy.copy(valid_report):
                        del valid_report[k]
            if args.max_train_time > -1:
                if (
                    'train_time' in valid_report
                    and valid_report['train_time'] > args.max_train_time
                ):
                    for k in copy.copy(valid_report):
                        del valid_report[k]

            # clean up metrics we don't care about if we

            metrics_keys |= set(valid_report.keys())
            nodelete_keys |= set(valid_report.keys())

            r = {**opt, **valid_report}
            r['valno'] = i
            r['stdout'] = stdout
            uid = sweep_folder.replace(root, '')
            if uid.startswith('/'):
                uid = uid[1:]
            r['uid'] = uid
            results.append(r)

    if len(results) == 0:
        return None

    if primary_metric is None:
        raise ValueError("No sort value!")

    if sort_direction is None:
        raise ValueError(
            'You must supply sort mode if you use a custom sorting metric.'
            ' Add either "-S min" or "-S max"'
        )

    results = pd.DataFrame(results)

    # Reorganize the columns such that those related to performance metrics are
    # on the rightmost side
    other_metrics = [x for x in metrics_keys if x != primary_metric]

    # filter any metrics the user doesn't want to see
    if metrics is not None:
        to_delete = []
        metrics = set([m.strip() for m in metrics])
        for om in other_metrics:
            if om.split('/')[-1] not in metrics:
                to_delete.append(om)
        for td in to_delete:
            other_metrics.remove(td)

    # Order the metrics by the similarity with the primary_metric so that the
    # hiplot displays fewer cross-lines
    non_multitask_metrics = [m for m in other_metrics if '/' not in m]
    multitask_metrics = [m for m in other_metrics if '/' in m]

    cols = sorted(
        [x for x in results.columns if x not in metrics_keys and x != 'stdout']
    )
    cols.extend(sorted(multitask_metrics))
    if non_multitask_metrics:
        cos_vec = (
            results[non_multitask_metrics]
            .apply(lambda x: (1 - cosine(x, results[primary_metric])), axis=0)
            .sort_values(ascending=True)
        )
        cols.extend(list(cos_vec.index))
    cols = cols + ['stdout', primary_metric]
    results = results.loc[:, cols]
    sortmode = sort_direction == "min"
    results = results.sort_values(
        [primary_metric, 'uid', 'valno'], ascending=[sortmode, True, False]
    )
    if best_only:
        results = results.groupby('uid').head(1)
        # don't need to remember which validation it is
        del results['valno']

    results = results.reset_index(drop=True)

    def _delete_columns(keys):
        colnames = set(results.columns)
        for k in keys:
            if k in results.columns:
                del results[k]
            for c in colnames:
                if c.endswith(f'/{k}') and c in results.columns:
                    del results[c]
                elif c.endswith('/total_train_updates') and c in results.columns:
                    # kill a bad metric we don't need everywhere
                    del results[c]

        for c in list(results.columns):
            if len(set(results[c])) == 1 and c not in nodelete_keys:
                del results[c]

    if filter_columns:
        _delete_columns(SKIP_KEYS)
    if ignore_fields:
        _delete_columns(ignore_fields)

    results.index = results.uid
    del results['uid']

    # group by multitasking
    if any('/' in c for c in results.columns):
        if allow_multitask:
            results.columns = pd.MultiIndex.from_tuples(results.columns.map(_split_key))
        else:
            new_cols = [c for c in results.columns if '/' not in c]
            results = results[new_cols]

    if max_rows is not None and max_rows > 0:
        results = results.head(max_rows)

    return results


def _last_sweep_dir():
    user = getpass.getuser()
    cp = f'/checkpoint/{user}/'
    timestamped = [f for f in os.listdir(cp) if f.startswith('20')]
    dirs = [os.path.join(cp, f) for f in timestamped]
    dirs = [d for d in dirs if os.path.isdir(d)]
    last_dir = sorted(dirs)[-1]
    subdirs = [os.path.join(last_dir, f) for f in os.listdir(last_dir)]
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    choice = max(subdirs, key=lambda d: os.stat(d).st_mtime)
    print(f"Guessing you wanted to check results of {choice}\n")
    return choice


def guess_sweep_folder() -> str:
    # first check the current folder
    cwd = os.getcwd()
    if '/private/home' in cwd:
        # we're probably in our home directory, guess the lalst checkpoint
        return _last_sweep_dir()
    if '/checkpoint' in cwd:
        # we're probably in a sweep folder
        return cwd
    # final guess, just go with last sweep directory
    return _last_sweep_dir()


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', default=None, nargs='?')
    parser.add_argument(
        '-W', '--no-wrap', '--nowrap', action='store_true', help="Don't wrap lines."
    )
    parser.add_argument(
        '--format',
        choices={'txt', 'html', 'vd', 'csv'},
        default='txt',
        help='Output format.',
    )
    parser.add_argument(
        '-s',
        '--sort-metric',
        default=None,
        help="Sort by this metric (default validation criterion).",
    )
    parser.add_argument(
        '-n', '--max', default=None, type=int, help='Maximum number of rows to show.'
    )
    parser.add_argument(
        '-m',
        '--metrics',
        default=None,
        help='Limit output to only these metrics (comma separated).',
    )
    parser.add_argument(
        '-c',
        '--max-column-width',
        default=MAX_COLUMN_WIDTH,
        type=int,
        help='Max column width for text mode.',
    )
    parser.add_argument(
        '--max-train-updates',
        default=-1,
        type=int,
        help='Do not consider reports with more train updates than this.',
    )
    parser.add_argument(
        '--max-train-time',
        default=-1,
        type=int,
        help='Do not consider reports with more train time than this.',
    )
    parser.add_argument(
        '-1',
        default=None,
        action='store_true',
        dest='justone',
        help='Only show the single best result.',
    )
    parser.add_argument('-S', '--sort-mode', choices={'min', 'max'}, default=None)
    parser.add_argument(
        '--no-subtasks', action='store_true', help="Don't display multitask metrics."
    )
    parser.add_argument(
        '--filter',
        default=None,
        type=str,
        help='Filter results by specific args; should be a comma separated list like:'
        '`--filter <arg1>=<value1>,<arg2>=<value2>`',
    )
    parser.add_argument(
        '--filter-delimiter',
        default=',',
        type=str,
        help='Delimiter for the --filter arg. Default is `,`.'
        '`--filter <arg1>=<value1>,<arg2>=<value2> --filter-delimiter ,`',
    )
    parser.add_argument(
        '--ignore-fields',
        default=None,
        type=str,
        help='Comma-separate list of columns to deliberately ignore'
        '`--ignore-fields <col1>,<col2>`',
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up the non-best models from a sweep folder. Requires confirmation.',
    )
    return parser


def _flat_string_results(results):
    """
    Rendered results that doesn't care about multicolumns.
    """
    line_width = shutil.get_terminal_size((88, 24)).columns
    sresult = results.to_string(
        na_rep='',
        float_format=float_formatter,
        line_width=line_width,
        index=True,
        max_colwidth=MAX_COLUMN_WIDTH,
        justify="justify-all",
    )
    sresult = sresult.replace(" 0.", "  .").replace(" -0.", "  -.")
    return sresult


def string_results(results):
    if not isinstance(results.columns, pd.MultiIndex):
        # no multiple metrics, just print:
        return _flat_string_results(results)

    final = []
    line_width = shutil.get_terminal_size((88, 24)).columns
    datasets = list(sorted(results.columns.levels[0]))
    if '' in datasets:
        datasets.remove('')
        datasets.append('')
    for k in datasets:
        final.append("\n")
        final.append(f"Dataset {k}" if k else "All")
        final.append("=" * line_width)
        final.append(_flat_string_results(results[k]))
    return '\n'.join(final)


def _recursive_find(root, is_sub_dir=False):
    # this could also be implemented with os.walk, but we chose not to so
    # we could shortcut
    if not is_sub_dir and not os.path.exists(os.path.join(root, 'grid.json')):
        print(
            f"{root} doesn't contain grid.json, so it probably isn't a sweep folder. "
            f"Cowardly exiting."
        )
        sys.exit(1)

    for subitem in os.listdir(root):
        complete = os.path.join(root, subitem)
        if subitem == 'ParlAI':
            continue
        elif os.path.isdir(complete):
            for x in _recursive_find(complete, is_sub_dir=True):
                yield x
        elif subitem == MODEL_FILENAME or subitem == CHECKPOINT_FILENAME:
            yield os.path.abspath(complete)


def cleanup_files(args, results):
    """
    Delete model files that are no longer necessary, based on sweep-results.
    """
    best_row = results.iloc[0]
    metric = results.columns[-1]
    value = best_row[metric]
    value = f'{value:.5g}'

    # complications below are due to handling multitasking
    if isinstance(metric, tuple):
        metric = " ".join(metric) if metric[0] != '' else metric[1]
    if 'model_file' in best_row:
        best_model = best_row['model_file']
    else:
        best_model = best_row[('', 'model_file')]

    # normalize all paths to be absolute to avoid ambiguities
    best_model = os.path.abspath(best_model)
    files_to_delete = list(_recursive_find(args.folder))
    files_to_delete.remove(best_model)

    # count file sizes
    total_savings = 0
    for fn in files_to_delete:
        total_savings += os.path.getsize(fn)
    total_savings = total_savings / (1024 * 1024 * 1024)
    keep_size = os.path.getsize(best_model) / (1024 * 1024 * 1024)

    if not files_to_delete:
        print("Did not find any files to clean up. Maybe you already ran cleanup?")
        sys.exit(1)

    print(" *" * 27)
    print("  " * 10 + "W A R N I N G")
    print(" *" * 27)
    print("Will DELETE these files:")
    print("\n".join(f"    {f}" for f in files_to_delete))
    print()
    print(f"Will keep this model with  ** {metric} = {value} **\n    {best_model}")
    print(f"Savings: {total_savings:.1f}gb   Keeping: {keep_size:.2f}gb")
    print()
    print("FYI: You should only run this on a sweep that is finished!")
    response = input(f"Type 'keep {value}' to continue: ")
    if response != f'keep {value}':
        print(f"You typed '{response}' instead of 'keep {value}'. Cowardly exiting.")
        sys.exit(1)

    for ftd in files_to_delete:
        print(f"Deleting {ftd}...")
        os.unlink(ftd)


def main():
    args = setup_args().parse_args()

    global MAX_COLUMN_WIDTH
    MAX_COLUMN_WIDTH = args.max_column_width

    if args.folder is None:
        args.folder = guess_sweep_folder()

    if args.filter is not None:
        args.filter = [
            (kv.split('=')[0], '='.join(kv.split('=')[1:]))
            for kv in args.filter.split(args.filter_delimiter)
        ]

    if args.justone:
        args.max = 1

    results = load_sweep(
        args,
        args.folder,
        # for these next two options, we need different behavior for cleanup
        best_only=args.cleanup,  # best_only True gives one score per model
        filter_columns=not args.cleanup,  # need to keep model_file column
        sort_metric=args.sort_metric,
        sort_mode=args.sort_mode,
        max_rows=args.max,
        metrics=args.metrics.split(',') if args.metrics else None,
        allow_multitask=(not args.no_subtasks and args.format != 'vd'),
        filter_opt=args.filter,
        ignore_fields=args.ignore_fields.split(',') if args.ignore_fields else None,
    )
    if results is None:
        print("Sorry, you don't have any results yet!")
        sys.exit(1)

    if args.cleanup:
        return cleanup_files(args, results)

    with pd.option_context('display.max_rows', None):
        if args.format == 'html':
            print(results.to_html(na_rep='', float_format=float_formatter))
        elif args.format == 'txt':
            print(string_results(results.groupby('uid').head(1)))

            best_row = results.iloc[0]
            best_key = results.index[0]
            if 'stdout' in results.columns:
                best_model = os.path.join(args.folder, best_row.stdout)
                last_5 = (
                    results.loc[[best_key]]
                    .sort_values('valno', ascending=False)
                    .head(5)
                )
            else:
                best_model = os.path.join(args.folder, best_row[('', 'stdout')])
                last_5 = (
                    results.loc[[best_key]]
                    .sort_values(('', 'valno'), ascending=False)
                    .head(5)
                )

            metric = results.columns[-1]
            best_score = float_formatter(best_row[metric])
            nice_metric = ' '.join(metric) if isinstance(metric, tuple) else metric
            last_scores = "  ".join(float_formatter(f) for f in last_5[metric])
            print()
            print(f"Best {nice_metric}: {best_score}")
            print(f"Log: {best_model}")
            print(f"Last {len(last_5)} validations (most recent first): {last_scores}")
        elif args.format == 'vd':
            try:
                import visidata
            except ImportError:
                raise ImportError('Please run `pip install visidata`')
            visidata.view_pandas(results)
        elif args.format == 'csv':
            results.to_csv(sys.stdout)


if __name__ == '__main__':
    main()
