import pandas as pd
from ast import literal_eval

cols_to_transform = ['seconds_to_incident_sequence', 'vehicles_sequence', 'events_sequence',
                     'train_kph_sequence', 'dj_ac_state_sequence', 'dj_dc_state_sequence']


def sort_by_first_column(row):
    combined = zip(*(row[col] for col in cols_to_transform))
    sorted_row = sorted(combined, key=lambda x: x[0])
    for i, col in enumerate(cols_to_transform):
        row[col] = [item[i] for item in sorted_row]
    return row


df = pd.read_csv("sncb_data_challenge.csv", sep = ';', index_col=0)


for col in cols_to_transform:
    df[col] = df[col].apply(literal_eval)


time_sorted_df = df.apply(sort_by_first_column, axis=1)

time_sorted_df.to_csv("time_sorted_table.csv", sep=';')
