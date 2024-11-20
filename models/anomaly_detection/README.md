# Anomaly detection using AutoEncoders
This model uses the AutoEncoders architecture to detect unusual/anomalous event sub-sequences that might indicate an incident.

To run the script:
```bash
python anomaly_detection.py --window <window size> --lag <lag size>
```
For example:
```bash
python anomaly_detection.py --window 30 --lag 3
```