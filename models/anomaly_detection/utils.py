import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def gen_seq(events, seconds, w, l):
    sequences = []
    anomaly_counts = []
    stop = False
    idx = 0
    cover = 0
    while not stop:
        if idx <= len(events) - w:
            seq = events[idx:idx+w]
            secs = seconds[idx:idx+w]
            anomaly_count = sum([1 for sec in secs if sec >= 0])
            if seq not in sequences:
                sequences.append(seq)
                anomaly_counts.append(anomaly_count)
                cover = idx + w - 1
            idx += l
        else:
            stop = True
    if cover < len(events) - 1 and len(events) >= w:
        seq = events[-w:]
        secs = seconds[-w:]
        sequences.append(seq)
        anomaly_counts.append(sum([1 for sec in secs if sec >= 0]))
    return sequences, anomaly_counts

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

def mse(A, B):
    return (np.square(A - B)).mean()

def plot_anomaly_scores(norm_errs, anom_errs, ambig_errs, filepath):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 8))

    axes[0].boxplot(np.array(norm_errs))
    axes[1].boxplot(np.array(anom_errs))
    axes[2].boxplot(np.array(ambig_errs))

    axes[0].set_title('Normal')
    axes[1].set_title('Anomalous')
    axes[2].set_title('Ambiguous')
    plt.savefig(f"{filepath}.png")
    plt.show()

def plot_point(df_mse, id, filepath=None):
    tmp_df = df_mse[df_mse['incident_id'] == id].reset_index(drop=True)
    ambig = tmp_df[tmp_df['class'] != 'normal'].index.to_list()[0]
    anom_count = tmp_df[tmp_df['class'] == 'anomalous'].index.to_list()
    if len(anom_count) > 0:
        anom = anom_count[0]
    else:
        anom = None

    fig = plt.figure(figsize=(10, 6))
    x = tmp_df.index.to_list()
    y = tmp_df.mse.to_list()
    plt.scatter(x, y)
    plt.vlines(x = ambig, ymin = 0, ymax = 80, colors = 'lightsalmon')
    if anom:
        plt.vlines(x = anom, ymin = 0, ymax = 80, colors = 'red')
    plt.xlabel('Event index')
    plt.ylabel('Reconstruction error')
    if filepath:
        plt.savefig(f"{filepath}.png")
    plt.show()

def plot_sns(df_mse, id, filepath=None):
    tmp_df = df_mse[df_mse['incident_id'] == id].reset_index(drop=True)
    ambig = tmp_df[tmp_df['class'] != 'normal'].index.to_list()[0]
    anom_count = tmp_df[tmp_df['class'] == 'anomalous'].index.to_list()
    if len(anom_count) > 0:
        anom = anom_count[0]
    else:
        anom = None

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tmp_df.reset_index(), x="index", y="mse", hue="class")
    if filepath:
        plt.savefig(f"{filepath}.png")
    plt.show()

def discounted_cumulative_gain(ranks):
    dcg = 0.0
    for rank in ranks:
        dcg = dcg + 1.0 / np.log2(rank + 1)
    return dcg

def normalized_discounted_cumulative_gain(ranks):
    num_gt = len(ranks)
    dcg = discounted_cumulative_gain(ranks)
    maxdcg = 0.0
    for i in range(1, num_gt + 1):
        maxdcg = maxdcg + 1.0 / np.log2(i + 1)
    return dcg / maxdcg

def eval_incident(df_mse, incident_id):
    incident_df = df_mse[df_mse['incident_id'] == incident_id]
    ranked_df = incident_df.sort_values(by='mse', ascending=False)
    a_list = incident_df[incident_df['class'] != 'normal']['num'].to_list()
    violators = ranked_df['num'].to_list()
    true_pos = list(set(violators) & set(a_list))
    true_pos_positions = [i+1 for i, x in enumerate(violators) if x in true_pos]
    ndcg = normalized_discounted_cumulative_gain(true_pos_positions)
    return ndcg