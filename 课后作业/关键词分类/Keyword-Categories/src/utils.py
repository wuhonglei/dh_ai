import time
import pathlib
import logging
import pandas as pd
import pygsheets

logger = logging.getLogger(__name__)

# file_path = pathlib.Path().resolve() / "config" / "google_sheet_key.json"
# gc = pygsheets.authorize(service_file=file_path)


def timer(function):
    def wrapper(*args, **kws):
        t_start = time.time()
        result = function(*args, **kws)
        t_end = time.time()
        t_count = t_end - t_start
        logger.info(
            f"<function {function.__qualname__}> - Time Coast: {t_count:.2f}s")
        return result

    return wrapper


@timer
def get_gsheet_df(sheet_name: str) -> pd.DataFrame:
    sht = gc.open_by_url(
        "https://docs.google.com/spreadsheets/d/1YRs61kY7gVRenlGn6yzMAiuPmaJ_NNmHxrWtXrQF8Ys/"
    )
    wks = sht.worksheet_by_title(sheet_name)
    return wks.get_as_df()


def get_df_from_xlsx(file_path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(file_path, sheet_name=sheet_name)


@timer
def update_gsheet_df(sheet_name: str, df: pd.DataFrame) -> None:
    sht = gc.open_by_url(
        "https://docs.google.com/spreadsheets/d/1YRs61kY7gVRenlGn6yzMAiuPmaJ_NNmHxrWtXrQF8Ys/"
    )
    wks = sht.worksheet_by_title(sheet_name)
    wks.clear()
    wks.set_dataframe(df, (1, 1))
