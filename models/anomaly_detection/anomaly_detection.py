import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers

from ast import literal_eval
from datetime import datetime
import os, sys, json

import utils

class AnomalyDetector:
    def __init__(self, w, l):
        print('[__init__] start')
        self.data_path = '../../data/time_sorted_table.csv'
        self.results_path = './results'
        self.w = w
        self.l = l
        self.vocab_path = './embeddings/metadata.tsv'
        self.vectors_path = './embeddings/vectors.tsv'
        timestamp = datetime.now()
        timestamp_fmt = timestamp.strftime("%Y%m%d_%H%M%S") # e.g. 20230320_154559
        self.result_id = f'{timestamp_fmt}_w{self.w}_l{self.l}'
        self.result_dir_path = f'{self.results_path}/{self.result_id}'
        if not os.path.exists(self.result_dir_path): 
            os.makedirs(self.result_dir_path)

        try:
            open(self.data_path)
        except Exception as err:
            print(f'Error opening input file: {err}')
            sys.exit(1)
        print('[__init__] end')

    def load_data(self):
        print('[load_data] start')
        df_og = pd.read_csv(self.data_path, delimiter=';', index_col=0)
        cols_to_transform = ['vehicles_sequence', 'events_sequence', 'seconds_to_incident_sequence',
                            'train_kph_sequence', 'dj_ac_state_sequence', 'dj_dc_state_sequence']
        for col in cols_to_transform:
            df_og[col] = df_og[col].apply(literal_eval)
        self.df_s = df_og[['incident_id', 'events_sequence', 'seconds_to_incident_sequence']]
        self.df_s['events_count'] = self.df_s.apply(lambda x: len(x.events_sequence), axis=1)
        print('[load_data] end')

    def generate_seqs(self):
        print('[generate_seqs] start')
        all_seqs = []
        all_counts = []
        all_ids = []
        all_nums = []
        skip = 0
        for _, row in self.df_s.iterrows():
            if row.events_count >= self.w:
                seqs, counts = utils.gen_seq(row.events_sequence, row.seconds_to_incident_sequence, self.w, self.l)
                all_seqs.append(seqs)
                all_counts.append(counts)
                all_ids.append([row.incident_id] * len(counts))
                all_nums.append(list(range(len(counts))))
            else:
                skip += 1
        col_names = [f't0{i}' if i < 10 else f't{i}' for i in range(self.w)]

        all_seqs_flat = [j for sub in all_seqs for j in sub]
        all_counts_flat = [j for sub in all_counts for j in sub]
        all_ids_flat = [j for sub in all_ids for j in sub]
        all_nums_flat = [j for sub in all_nums for j in sub]
        self.df_seq = pd.DataFrame(np.array(all_seqs_flat), columns=col_names)
        self.df_seq['anom_count'] = all_counts_flat
        self.df_seq['incident_id'] = all_ids_flat
        self.df_seq['num'] = all_nums_flat
        print('[generate_seqs] end')

    def classify_seqs(self):
        print('[classify_seqs] start')
        def classify(row, w):
            if row.anom_count == 0:
                return 'normal'
            if row.anom_count == w:
                return 'anomalous'
            if row.anom_count > 0 and row.anom_count < w:
                return 'ambiguous'
        self.df_seq['class'] = self.df_seq.apply(lambda x: classify(x, self.w), axis=1)

        self.df_norm = self.df_seq[self.df_seq['class'] == 'normal']
        self.df_anom = self.df_seq[self.df_seq['class'] == 'anomalous']
        self.df_ambig = self.df_seq[self.df_seq['class'] == 'ambiguous']
        print('[classify_seqs] end')

    def encode_seqs(self):
        print('[encode_seqs] start')
        self.vocab = pd.read_csv(self.vocab_path, delimiter='\t', header=None)
        self.vocab.columns = ['word']
        self.vectors = pd.read_csv(self.vectors_path, delimiter='\t', header=None)
        vocab_lookup = dict()
        for idx, row in self.vocab.iterrows():
            vocab_lookup[row['word']] = self.vectors.iloc[idx].values
        df_norm_tmp = self.df_norm.drop(columns=['anom_count', 'incident_id', 'num', 'class'], axis=1)
        df_anom_tmp = self.df_anom.drop(columns=['anom_count', 'incident_id', 'num', 'class'], axis=1)
        df_ambig_tmp = self.df_ambig.drop(columns=['anom_count', 'incident_id', 'num', 'class'], axis=1)
        self.encoded_normal = utils.encode_seqs(df_norm_tmp, vocab_lookup)
        self.encoded_anom = utils.encode_seqs(df_anom_tmp, vocab_lookup)
        self.encoded_ambig = utils.encode_seqs(df_ambig_tmp, vocab_lookup)
        print('[encode_seqs] end')

    def create_model(self):
        print('[create_model] start')
        seq_len = self.w
        embedding_dim = self.vectors.shape[1]
        self.model = models.Sequential()
        self.model.add(layers.LSTM(100, activation='relu',
                            input_shape=(seq_len, embedding_dim),
                            kernel_initializer='glorot_uniform'))
        self.model.add(layers.RepeatVector(seq_len))
        self.model.add(layers.LSTM(100, activation='relu', return_sequences=True))
        self.model.add(layers.TimeDistributed(layers.Dense(embedding_dim)))
        optimizer = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6, clipvalue=1.0)
        self.model.compile(optimizer=optimizer, loss='mse')
        print('[create_model] end')

    def train_model(self):
        print('[train_model] start')
        epochs = 5
        self.model.fit(self.encoded_normal, self.encoded_normal, epochs=epochs)
        print('[train_model] end')

    def get_anomaly_scores(self):
        print('[get_anomaly_scores] start')
        self.y_norm = self.model.predict(self.encoded_normal)
        self.y_anom = self.model.predict(self.encoded_anom)
        self.y_ambig = self.model.predict(self.encoded_ambig)
        
        self.norm_errs = []
        for i in range(len(self.y_norm)):
            self.norm_errs.append(utils.mse(self.y_norm[i], self.encoded_normal[i]))

        self.anom_errs = []
        for i in range(len(self.y_anom)):
            self.anom_errs.append(utils.mse(self.y_anom[i], self.encoded_anom[i]))

        self.ambig_errs = []
        for i in range(len(self.y_ambig)):
            self.ambig_errs.append(utils.mse(self.y_ambig[i], self.encoded_ambig[i]))
        
        # Plot anomaly scores by class
        filepath = f'{self.result_dir_path}/anomaly_scores_plot'
        utils.plot_anomaly_scores(self.norm_errs, self.anom_errs, self.ambig_errs, filepath)
        print('[get_anomaly_scores] end')

    def evaluate(self):
        print('[evaluate] start')
        self.df_norm['mse'] = self.norm_errs
        self.df_anom['mse'] = self.anom_errs
        self.df_ambig['mse'] = self.ambig_errs
        self.df_mse = pd.concat([self.df_norm, self.df_anom, self.df_ambig])
        self.df_mse = self.df_mse.sort_values(by=['incident_id', 'num'])
        self.NDCGs = dict()
        for incident_id in self.df_mse['incident_id'].unique():
            incident_df = self.df_mse[self.df_mse['incident_id'] == incident_id]
            non_norm_count = sum([1 for x in incident_df["class"].to_list() if x != "normal"])
            norm_count = sum([1 for x in incident_df["class"].to_list() if x == "normal"])
            if non_norm_count != 0 and norm_count != 0:
                self.NDCGs[incident_id] = utils.eval_incident(self.df_mse, incident_id)
            elif non_norm_count == 0:
                print(f'{incident_id}: non-norm count = 0')
            elif norm_count == 0:
                print(f'{incident_id}: norm count = 0')
        self.ndcg_mean = np.mean(list(self.NDCGs.values()))
        self.ndcg_median = np.median(list(self.NDCGs.values()))
        print('Normalized discounted cumulative gain (NDCG) score:')
        print(f'\tMean: {self.ndcg_mean}')
        print(f'\tMedian: {self.ndcg_median}')
        print('[evaluate] end')
    
    def save_results(self):
        print('[save_results] start')
        self.model.save_weights(f'{self.result_dir_path}/model.weights.h5')
        self.df_seq.to_csv(f'{self.result_dir_path}/df_seq.csv')
        self.df_mse.to_csv(f'{self.result_dir_path}/df_mse.csv')
        NDCGs_tmp = dict()
        for key, value in self.NDCGs.items():
            NDCGs_tmp[str(key)] = value
        with open(f'{self.result_dir_path}/NDCGs_all.json', 'w') as fp:
            json.dump(obj=NDCGs_tmp, fp=fp, indent=4)
        score = {
            'ndcg_mean': self.ndcg_mean,
            'ndcg_median': self.ndcg_median
        }
        with open(f'{self.result_dir_path}/NDCG.json', 'w') as fp:
            json.dump(obj=score, fp=fp, indent=4)
        print('[save_results] end')