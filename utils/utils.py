import sys
import pandas as pd 
from ast import literal_eval

array_cols = ['seconds_to_incident_sequence', 'vehicles_sequence', 'events_sequence',
                     'train_kph_sequence', 'dj_ac_state_sequence', 'dj_dc_state_sequence']


def arrify_string_columns(df: pd.DataFrame): 
    for col in array_cols:
        df[col] = df[col].apply(literal_eval)


def deep_copy(df: pd.DataFrame, additional_columns = []) -> pd.DataFrame: 
    cols_to_copy = list.copy(array_cols)
    copy_df = df.copy(deep=True)
    cols_to_copy.extend(additional_columns)
    cols_to_copy = set(cols_to_copy)

    for col in cols_to_copy:
        if col in df: 
            copy_df[col] = df[col].apply(list.copy)

    return copy_df
