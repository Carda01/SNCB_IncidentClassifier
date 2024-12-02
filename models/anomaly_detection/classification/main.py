import argparse
import lstm_classifier as lc

parser = argparse.ArgumentParser(description="Multi-class classifier using LSTM")
parser.add_argument('--normal_only', '-n', help='Whether or not to use only normal subsequences', required=True, type=int, choices=[0, 1])
parser.add_argument('--percent', '-p', help='Which percentage to use', required=True, type=int, choices=[10, 15, 20, 30, 50])
args = parser.parse_args()

def main():
    normal_only = True if args.normal_only == 1 else False
    LC = lc.LSTMClassifier(normal_only=normal_only, perc=args.percent)
    LC.load_data()
    LC.encode_seqs()
    LC.prepare_train_data()
    LC.create_model()
    LC.train_model()
    LC.evaluate()
    LC.save_results()

if __name__ == '__main__':
    main()