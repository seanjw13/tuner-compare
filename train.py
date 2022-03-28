
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn import metrics
import joblib

# Pass in environment variables and hyperparameters
parser = argparse.ArgumentParser()

# Hyperparameters
parser.add_argument("--estimators", type=int, default=15)

# sm_model_dir: model artifacts stored here after training
parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
parser.add_argument("--model_dir", type=str)
parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

args, _ = parser.parse_known_args()
estimators = args.estimators
model_dir = args.model_dir
sm_model_dir = args.sm_model_dir
training_dir = args.train

print(os.listdir(training_dir))
print()
# Create ensemble model
#model = gbr(learning_rate=params['lr'],
#            n_estimators=int(params['n_est']),
#            max_depth=int(params['max_depth']))

#model.fit(train_x, train_y)
#eval_results = model.score(test_x, test_y)

# Save model
#joblib.dump(regressor, os.path.join(args.sm_model_dir, "model.joblib"))