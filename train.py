from sklearn.ensemble import GradientBoostingRegressor as gbr

def train_model(train_data, test_data, n_est=10, max_depth=5, lr=.01, 
                label="quality"):

    # Create ensemble model
    model = gbr(learning_rate=lr,
                n_estimators=n_est,
                max_depth=max_depth)

    train_y = train_data.pop(label)
    test_y = test_data.pop(label)

    print("""Training a new model with a learning rate of {}, {} estimators,
          and a max_depth of {}""".format(lr, n_est, max_depth))

    model.fit(train_data, train_y)
    
    return model.score(test_data, test_y)