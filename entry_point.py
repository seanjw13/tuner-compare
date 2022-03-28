import argparse, os

def parse_sage_args():
    # Pass in environment variables and hyperparameters
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--estimators", type=int, default=15)

    # sm_model_dir: model artifacts stored here after training
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":

    args = parse_sage_args()

    # Read locally training files
    train_folder = os.getenv("SM_CHANNEL_TRAINING")
    estimators = args.estimators
    model_dir = args.model_dir
    sm_model_dir = args.sm_model_dir
    training_dir = args.train

    print("Content of training_dir: {}\n".format(os.listdir(training_dir)))
    print("Chosen hyperparameters are: {}\n".format(estimators))
    print("The local directory that holds the training data is: {}\n".format(train_folder))
    print("Content of training_dir: {}\n".format(os.listdir(train_folder)))