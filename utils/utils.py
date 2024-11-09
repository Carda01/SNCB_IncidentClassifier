import pandas as pd 
from ast import literal_eval

def arrify_string_columns(df: pd.DataFrame): 
    cols_to_transform = ['seconds_to_incident_sequence', 'vehicles_sequence', 'events_sequence',
                         'train_kph_sequence', 'dj_ac_state_sequence', 'dj_dc_state_sequence']

    for col in cols_to_transform:
        df[col] = df[col].apply(literal_eval)
