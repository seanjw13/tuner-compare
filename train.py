from sklearn.ensemble import GradientBoostingRegressor as gbr

def train_model(num_est=10, max_depth=5, lr=.01):

    # Create ensemble model
    model = gbr(learning_rate=lr,
                n_estimators=num_est,
                max_depth=max_depth)

    model.fit(train_x, train_y)
    eval_results = model.score(test_x, test_y)

# Save model
#joblib.dump(regressor, os.path.join(args.sm_model_dir, "model.joblib"))