def optimize_models(X_train, X_val, y_train, y_val, n_trials=20):

    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'n_jobs': -1}
        
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return -mean_squared_error(y_val, preds, squared=False)  # minimize RMSE

    def objective_et(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'n_jobs': -1}
        
        model = ExtraTreesRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return -mean_squared_error(y_val, preds, squared=False)

    def objective_cat(trial):
        params = {
            'depth': trial.suggest_int('depth', 5, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'iterations': trial.suggest_int('iterations', 300, 1000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 2.0),
            'thread_count': 8,
            'verbose': 0}
        
        model = CatBoostRegressor(**params, random_state=42)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        preds = model.predict(X_val)
        return -mean_squared_error(y_val, preds, squared=False)

    models = {
        "RandomForest": (objective_rf, RandomForestRegressor),
        "ExtraTrees": (objective_et, ExtraTreesRegressor),
        "CatBoost": (objective_cat, CatBoostRegressor)}

    results = []
    best_models = {}
    sampler = optuna.samplers.TPESampler()
    total = len(models)

    for idx, (name, (objective, cls)) in enumerate(models.items(), 1):
        print(f"[{idx}/{total}] Tuning model: {name}...")
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"Best params for {name}: {best_params}")

        final_model = cls(**best_params, random_state=42)
        final_model.fit(X_train, y_train)
        preds = final_model.predict(X_val)

        rmse = mean_squared_error(y_val, preds, squared=False)
        mse  = mean_squared_error(y_val, preds)
        mae  = mean_absolute_error(y_val, preds)
        r2   = r2_score(y_val, preds)

        results.append({
            "model": name,
            "RMSE": rmse,
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "Best Params": best_params})

        best_models[name] = final_model

    df_result = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
    return df_result, best_models
