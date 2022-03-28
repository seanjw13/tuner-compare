import argparse, os
import pandas as pd

from train import train_model

def parse_sage_args():
    # Pass in environment variables and hyperparameters
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--n_est", type=int, default=15)
    parser.add_argument("--max_depth", type=int, default=5)

    # sm_model_dir: model artifacts stored here after training
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    args, _ = parser.parse_known_args()

    return args

def load_data(training_dir, test_dir):
    
    train_data = pd.read_csv(training_dir + "/train.csv")
    test_data = pd.read_csv(test_dir + "/test.csv")

    return train_data, test_data

if __name__ == "__main__":

    args = parse_sage_args()

    # Read local data    
    train_data, test_data = load_data(args.train, args.test)

    val_metric = train_model(train_data, test_data, lr=args.lr, 
                             n_est=args.n_est, max_depth=args.max_depth)

    # Print out validation metric so SageMaker hyperparameter tuner can parse
    # the logs and read it.
    print("Model validation metric R2:{0:.4f};".format(val_metric))