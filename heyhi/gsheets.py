# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities to export run resuls to google sheets.

See https://erikrood.com/Posts/py_gsheets.html for setup.

The OAuth creditinal file should be saved as client_secret.json.
Then run this module to finish/test authorization.
"""
import datetime
import logging
import string
import warnings

try:
    import pygsheets
except ImportError:
    pygsheets = None


TB_ADDRESS = "localhost:8888"
NOTEBOOK_ADDRESS = "localhost:8892"


def get_timezone_offset_hours():
    return round((datetime.datetime.now() - datetime.datetime.utcnow()).total_seconds() / 3600)


def save_pandas_table(dataframes, project_name, table_name, offset=0, start=10, viz_port=None):
    """Saves/updates a sheet with the dataframe."""
    if pygsheets is None:
        warnings.warn("Failed to import 'pygsheets'. save_pandas_table will do nothing")
        return
    try:
        client = pygsheets.authorize()
    except:
        logging.exception("Failed to authorize in google drive")
        return
    if project_name not in client.spreadsheet_titles():
        client.create(project_name)
    sheet = client.open(project_name)
    if table_name not in (wsh.title for wsh in sheet.worksheets()):
        created = True
        sheet.add_worksheet(table_name)
    else:
        created = False
    worksheet = sheet.worksheet_by_title(table_name)
    worksheet.update_value((1, 1), f"Last update:")
    update_cell = "$B$1"
    now_date = datetime.datetime.now().isoformat().replace("T", " ")
    worksheet.update_value((1, 2), now_date)
    if created:
        offset_hours = get_timezone_offset_hours()
        hours_passed = f"((TO_PURE_NUMBER(NOW()) - TO_PURE_NUMBER($B$1)) * 24 + ({offset_hours}))"
        worksheet.update_value((1, 3), f"={hours_passed} * 60")
        worksheet.update_value((1, 4), "mins ago")
    if not isinstance(dataframes, (tuple, list)):
        dataframes = [dataframes]
    for dataframe in dataframes:
        if "folder" in dataframe.columns:
            dataframe = dataframe.copy()
            dataframe["logs"] = [
                f'=HYPERLINK("http://{NOTEBOOK_ADDRESS}/tree/root{folder}", "folder")'
                for folder in dataframe.folder
            ]
            if "job_id" in dataframe.columns:
                dataframe["err_logs"] = [
                    f'=HYPERLINK("http://{NOTEBOOK_ADDRESS}/edit/root{folder}/slurm/{job_id}_0_log.err", "log")'
                    for folder, job_id in zip(dataframe.folder, dataframe.job_id)
                ]
            if viz_port:
                dataframe["viz"] = [
                    f'=HYPERLINK("http://localhost:{viz_port}/?game={folder}", "Open Viz")'
                    for folder in dataframe.folder
                ]
            dataframe["tb"] = [
                f'=HYPERLINK("http://{TB_ADDRESS}/?logs={folder}", "Open TB")'
                for folder in dataframe.folder
            ]
        worksheet.set_dataframe(dataframe, (start, 1 + offset))
        if "status" in dataframe.columns and created:
            _add_status_conditional_coloring(
                sheet,
                worksheet,
                list(dataframe.columns).index("status") + offset,
                start,
                len(dataframe.index) + 100,
            )
        if "status" in dataframe.columns and "last" in dataframe.columns and created:
            _add_last_conditional_coloring(
                sheet,
                worksheet,
                list(dataframe.columns).index("status") + offset,
                list(dataframe.columns).index("last") + offset,
                update_cell,
                start,
                len(dataframe.index) + 100,
            )
        start += len(dataframe.index) + 5


def _add_status_conditional_coloring(sheet, worksheet, col, start_row, length):
    boolean_rules = [
        {
            "condition": {"type": "TEXT_CONTAINS", "values": [{"userEnteredValue": "DONE"}]},
            "format": {"backgroundColor": {"red": 0.85, "green": 0.92, "blue": 0.83}},
        },
        {
            "condition": {"type": "TEXT_CONTAINS", "values": [{"userEnteredValue": "RUNN"}]},
            "format": {"backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.8}},
        },
        {
            "condition": {"type": "TEXT_CONTAINS", "values": [{"userEnteredValue": "DEAD"}]},
            "format": {"backgroundColor": {"red": 0.99, "green": 0.9, "blue": 0.8}},
        },
    ]
    for i, rule in enumerate(boolean_rules):
        request = {
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [
                        {
                            "sheetId": worksheet.id,
                            "startColumnIndex": col,
                            "endColumnIndex": col + 1,
                            "startRowIndex": start_row,
                            "endRowIndex": start_row + length,
                        }
                    ],
                    "booleanRule": rule,
                },
                "index": 100 + i,
            }
        }
        sheet.custom_request(request, "")


def _add_last_conditional_coloring(
    sheet, worksheet, status_col, last_col, update_cell, start_row, length
):
    """Mark running jobs without updates."""
    status_col_name = string.ascii_uppercase[status_col]
    last_col_name = string.ascii_uppercase[last_col]

    time_different_rule = f"(TO_PURE_NUMBER({update_cell}) - TO_PURE_NUMBER({last_col_name}{start_row + 1})) * 24 > 1"
    is_running_rule = f'REGEXMATCH({status_col_name}{start_row + 1}, "RUNN")'
    boolean_rules = [
        {
            "condition": {
                "type": "CUSTOM_FORMULA",
                "values": [
                    {"userEnteredValue": f"=AND({time_different_rule}, {is_running_rule})"}
                ],
            },
            "format": {"backgroundColor": {"red": 0.99, "green": 0.9, "blue": 0.8}},
        }
    ]
    for i, rule in enumerate(boolean_rules):
        request = {
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [
                        {
                            "sheetId": worksheet.id,
                            "startColumnIndex": last_col,
                            "endColumnIndex": last_col + 1,
                            "startRowIndex": start_row,
                            "endRowIndex": start_row + length,
                        }
                    ],
                    "booleanRule": rule,
                },
                "index": 100 + i,
            }
        }
        sheet.custom_request(request, "")


if __name__ == "__main__":
    pygsheets.authorize()
