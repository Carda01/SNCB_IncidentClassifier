import argparse
import anomaly_detection as ad

parser = argparse.ArgumentParser(description="Anomaly detection model with AutoEncoders")
parser.add_argument('--window', '-w', help='Window size', required=True, type=int)
parser.add_argument('--lag', '-l', help="Size of lag step", required=True, type=int)
args = parser.parse_args()

def main():
    AD = ad.AnomalyDetector(w=args.window, l=args.lag)
    AD.load_data()
    AD.generate_seqs()
    AD.classify_seqs()
    AD.encode_seqs()
    AD.create_model()
    AD.train_model()
    AD.get_anomaly_scores()
    AD.evaluate()
    AD.save_results()

if __name__ == "__main__":
    main()