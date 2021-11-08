# 调参
from sklearn.model_selection import GridSearchCV
#####################################################
logging.warning("Begin Model.")
result = {}  # 保存一些结果
# lightGBM 模型
from lightgbm import LGBMClassifier

# 5.1 模型定义
param_grid = {
    'n_estimators': [10, 100, 1000, 10000]
}
grid_search = GridSearchCV(
    xgb.XGBClassifier(
        learning_rate=0.1,
    ),
    param_grid,
    scoring='roc_auc', n_jobs=4, cv=5)
grid_search.fit(X_train, Y_train)
result['acc0'] = grid_search.score(X_test, Y_test)
result['best_params0'] = grid_search.best_params_
result['best_score0'] = grid_search.best_score_
result['n_estimators'] = grid_search.best_params_['n_estimators']
fixed_params = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': True,
    'boosting': 'gbdt',
    'num_boost_round': 300,
    'early_stopping_rounds': 30
}
search_params = {
    'learning_rate': 0.4,
    'max_depth': 15,
    'num_leaves': 20,
    'feature_fraction': 0.8,
    'subsample': 0.2
}
model = LGBMClassifier(
    #     objective='binary',
    #     metric='auc',
    #     is_unbalance=True,
    #     boosting='gbdt',
    #     num_boost_round=300,
    #     early_stopping_rounds=30,

    #     learning_rate=0.4,
    #     max_depth=15,
    #     num_leaves=20,
    #     feature_fraction=0.8,
    #     subsample=0.2,

    verbose=10
)

# 5.2 模型训练
import time

time_start = time.time()
model.fit(X_train, Y_train)
time_end = time.time()
print('time cost', time_end - time_start, 's')
# 5.3 模型评估
result['score_train'] = model.score(X_train, Y_train)
result['score_test'] = model.score(X_test, Y_test)
# 5.4 模型保存
joblib.dump(model, f'./model/lightGBM_(%f, %f, %f).model' %
            (test_size, result['score_train'], result['score_test']))
# 5.5 结果保存
print(result)
# with open(result_path, 'w') as fp:
#     json.dump(result, fp)
#####################################################
