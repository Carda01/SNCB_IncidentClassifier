import sys
import os
import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
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


class IncidentStats:
    def __init__(self, original_df):
        self.data_folder = os.path.join("..", "data")
        self.original_df = deep_copy(original_df)

        data_dir = os.path.join(self.data_folder, 'nuts/')
        path_rg = data_dir + "NUTS_RG_01M_2021_3035_LEVL_0.json"
        gdf_rg = gpd.read_file(path_rg)
        gdf_rg.crs = "EPSG:3035"
        gdf_rg = gdf_rg.to_crs("EPSG:3857")

        self.belgium = gdf_rg[gdf_rg.CNTR_CODE == "BE"]
        self.europe = gdf_rg[gdf_rg.CNTR_CODE.isin(["BE", "DE", "FR", "LU", "NL"])]
        cmap = ListedColormap(sns.color_palette('Paired', 12))
        incident_type_counts = Counter(original_df.incident_type)

        sorted_incident_types = sorted(incident_type_counts, key=incident_type_counts.get)
        self.color_lookup = {}
        for i, incident_type in enumerate(sorted_incident_types):
            self.color_lookup[incident_type] = cmap(i)




    def get_geospatial_summary(self, df):
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.approx_lon, df.approx_lat),
            crs="EPSG:4326"
        )

        gdf = gdf.to_crs("EPSG:3857")

        filtered_gdf = gdf[gdf.approx_lat>45]
        outside_belgium_tuples = gdf[gdf.approx_lat<=45]

        fig, ax = plt.subplots(figsize=(10, 10))
        self.europe.plot(ax=ax, color="lightgrey", edgecolor = "black")
        self.belgium.plot(ax=ax, color="bisque", edgecolor="black")

        incident_type_counts = Counter(gdf.incident_type)

        sorted_incident_type = sorted(incident_type_counts, key=incident_type_counts.get)

        ax.set_xlim(self.belgium.total_bounds[[0, 2]])
        ax.set_ylim(self.belgium.total_bounds[[1, 3]])

        for i, label in enumerate(sorted_incident_type[::-1]):
            subset = filtered_gdf[filtered_gdf.incident_type == label]
            subset.plot(ax=ax, color=self.color_lookup[label], label=f'{label}', markersize=35, edgecolor='k')

        plt.legend(title='Incident type', loc='lower left')
        plt.axis('off')


    def get_stats_incidents(self, df):
        stats_incidents = df.groupby('incident_type').size().reset_index(name='count')
        stats_incidents['percentage'] = (stats_incidents['count'] / len(df)) * 100
        original_stats = self.original_df.groupby('incident_type').size().reset_index(name='count_original')
        stats_incidents = stats_incidents.merge(original_stats, how='inner', on='incident_type')
        stats_incidents['count_percentage_on_total'] = stats_incidents['count']/stats_incidents['count_original'] * 100
        stats_incidents = stats_incidents.drop('count_original', axis=1)
        
        return stats_incidents