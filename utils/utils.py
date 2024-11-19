import sys
import pandas as pd 
from math import log
from collections import Counter
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


def get_frequency(list):
    element_counts = Counter(list)
    total_elements = len(list)
    return {element: count / total_elements for element, count in element_counts.items()}


def calculate_tfidf(events_column, dumping_function):
    events = []
    count = Counter()
    for events in events_column:
        count.update(set(events))

    number_of_documents_with_event = dict(count)
    number_of_rows = len(events_column)

    frequencies = events_column.apply(get_frequency)

    tfidf = {}
    for event in number_of_documents_with_event.keys(): 
        idf = log(number_of_rows/ number_of_documents_with_event[event])
        tfidf[event] = frequencies.apply(lambda x: idf * dumping_function(x.get(event, 0)))

    return tfidf