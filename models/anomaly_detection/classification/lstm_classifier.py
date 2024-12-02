import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

from ast import literal_eval
from datetime import datetime
import os, sys, json

class LSTMClassifier:
    def __init__(self, normal_only, perc):
        print('[__init__] start')
        if normal_only:
            self.data_dir = './data_norm'
        else:
            self.data_dir = './data_all'
        self.perc = perc
        self.vocab_path = '../embeddings/metadata.tsv'
        self.vectors_path = '../embeddings/vectors.tsv'
        self.data_path = f'{self.data_dir}/top_seq_{self.perc}.csv'
        self.results_path = './results'
        timestamp = datetime.now()
        timestamp_fmt = timestamp.strftime("%Y%m%d_%H%M%S") # e.g. 20230320_154559
        norm_str = 'norm' if normal_only else 'all'
        self.result_id = f'{timestamp_fmt}_{self.perc}_{norm_str}'
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
        self.df = pd.read_csv(self.data_path, index_col=0)
        self.df_seq = self.df.drop(columns=['anom_count', 'incident_id', 'num', 'class', 'mse'], axis=1)
        self.df_og = pd.read_csv('../../../data/time_sorted_table.csv', delimiter=';', index_col=0)
        self.incident_type_dict = dict()
        for _, row in self.df_og.iterrows():
            self.incident_type_dict[row['incident_id']] = row['incident_type']
        self.df['incident_type'] = self.df['incident_id'].map(self.incident_type_dict)
        print('[load_data] end')

    def encode_seqs(self):
        print('[encode_seqs] start')
        def encode_seqs(df, vocab_lookup):
            encoded = []
            for _, row in df.iterrows():
                seq = []
                for i, step in enumerate(row.values):
                    try:
                        seq.append(vocab_lookup[str(step)])
                    except:
                        print(f'Unknown: {step}')
                        seq.append(vocab_lookup['[UNK]'])
                encoded.append(seq)
            print(len(encoded), len(encoded[0]))
            return np.array(encoded)
        self.vocab = pd.read_csv(self.vocab_path, delimiter='\t', header=None)
        self.vocab.columns = ['word']
        self.vectors = pd.read_csv(self.vectors_path, delimiter='\t', header=None)
        vocab_lookup = dict()
        for idx, row in self.vocab.iterrows():
            vocab_lookup[row['word']] = self.vectors.iloc[idx].values
        self.df_encoded = encode_seqs(self.df_seq, vocab_lookup)
        print('[encode_seqs] end')

    def prepare_train_data(self):
        print('[prepare_train_data] start')
        self.y = np.array(self.df['incident_type'].to_list())
        self.y_encoder = LabelEncoder()
        self.y_encoded = self.y_encoder.fit_transform(self.y)

        self.X = self.df_encoded
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y_encoded, test_size=0.3, random_state=42)

        self.X_scaler = StandardScaler()
        X_train_reshaped = self.X_train.reshape(-1, self.X_train.shape[-1])
        self.X_train_scaled = self.X_scaler.fit_transform(X_train_reshaped)
        self.X_train_scaled = self.X_train_scaled.reshape(self.X_train.shape)
        
        X_test_reshaped = self.X_test.reshape(-1, self.X_test.shape[-1])
        self.X_test_scaled = self.X_scaler.transform(X_test_reshaped)
        self.X_test_scaled = self.X_test_scaled.reshape(self.X_test.shape)
        print('[prepare_train_data] end')

    def create_model(self):
        print('[create_model] start')
        seq_length = self.X.shape[1]
        embedding_dim = self.X.shape[2]
        num_classes = len(np.unique(self.y))

        event_input = layers.Input(shape=(seq_length, embedding_dim), name='event_input')
        lstm = layers.LSTM(units=100, dropout=0.3,
                        kernel_initializer='glorot_uniform')(event_input)
        x = layers.Dense(64, activation='relu')(lstm)
        output = layers.Dense(num_classes, activation='softmax')(x)

        self.model = keras.models.Model(inputs=[event_input], outputs=output)
        opt = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6, clipvalue=1.0)
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print('[create_model] end')

    def visualize_model(self):
        return keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)

    def train_model(self):
        print('[train_model] start')
        epochs = 100
        self.history = self.model.fit({"event_input": self.X_train_scaled},
                                      self.y_train,
                                      batch_size=36, verbose=1, epochs=epochs,
                                      validation_split=0.2)
        print('[train_model] end')

    def evaluate(self):
        print('[evaluate] start')
        def get_incident_prediction(df, incident_id):
            df_incident = df[df['incident_id'] == incident_id]
            preds_lst = df_incident['prediction'].to_list()
            return max(set(preds_lst), key=preds_lst.count)
        X_reshaped = self.X.reshape(-1, self.X.shape[-1])
        self.X_scaled = self.X_scaler.transform(X_reshaped)
        self.X_scaled = self.X_scaled.reshape(self.X.shape)

        self.preds_full = self.model.predict(self.X_scaled)
        self.pred_labels_full = np.argmax(self.preds_full, axis=1)
        self.preds_og = self.y_encoder.inverse_transform(self.pred_labels_full)
        print('Evaluation on subsequence level')
        self.clf_report_sub = classification_report(self.y, self.preds_og, output_dict=True)
        print(classification_report(self.y, self.preds_og))

        self.df['prediction'] = self.preds_og
        self.incident_labels = []
        self.incident_preds = []
        for incident_id in self.df['incident_id'].unique():
            self.incident_labels.append(self.incident_type_dict[incident_id])
            self.incident_preds.append(get_incident_prediction(self.df, incident_id))
        print('\nEvaluation on incident level')
        self.clf_report_inc = classification_report(self.incident_labels, self.incident_preds, output_dict=True)
        print(classification_report(self.incident_labels, self.incident_preds))
        print('[evaluate] end')

    def save_results(self):
        print('[save_results] start')
        self.model.save_weights(f'{self.result_dir_path}/model.weights.h5')
        self.df.to_csv(f'{self.result_dir_path}/df_preds.csv')
        with open(f'{self.result_dir_path}/clf_report_sub.json', 'w') as fp:
            json.dump(obj=self.clf_report_sub, fp=fp, indent=4)
        with open(f'{self.result_dir_path}/clf_report_inc.json', 'w') as fp:
            json.dump(obj=self.clf_report_inc, fp=fp, indent=4)
        print('[save_results] end')