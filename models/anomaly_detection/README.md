# Anomaly detection using AutoEncoders
This model uses the AutoEncoders architecture to detect unusual/anomalous event sub-sequences that might indicate an incident. You can test the model using the command line or import the model to explore in a notebook.

## Command line
```bash
python main.py --window <window size> --lag <lag size>
```
For example:
```bash
python main.py --window 30 --lag 3
```

## Notebook
```Python
import anomaly_detection as ad

w = <window size>
l = <lag size>

AD = ad.AnomalyDetector(w=w, l=l)
AD.load_data()
AD.generate_seqs()
AD.classify_seqs()
AD.encode_seqs()
AD.create_model()
AD.train_model()
AD.get_anomaly_scores()
AD.evaluate()
AD.save_results()
```

If you're running on Google Colab, you need to navigate to this directory first.
```Python
path = <path to this directory>
%cd $path
import anomaly_detection as ad
...
```