"""运行 python 02_train_model.py；依赖 pandas numpy scikit-learn matplotlib。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / 'movies_clean.csv')

# 拆分多国家、多类型为独立指示特征。此处仅解析每行，不从全数据学习类别。
def features(row):
    result = {'release_year': float(row['release_year'])}
    result['rating=' + str(row['rating'])] = 1.0
    for col in ['country', 'listed_in']:
        for label in set(str(row[col]).split(',')):
            result[col + '=' + label.strip()] = 1.0
    return result

X = [features(row) for _, row in d.iterrows()]
y = d['duration_min'].to_numpy()
indices = np.arange(len(d))
train, test = train_test_split(indices, test_size=0.2, random_state=42)
X_train, X_test = [X[i] for i in train], [X[i] for i in test]
y_train, y_test = y[train], y[test]

# 编码器和标准化都在 Pipeline 内：每个交叉验证训练折分别拟合。
# 稀疏矩阵不能直接中心化，ElasticNet 的截距处理均值。
pipeline = Pipeline([
    ('encode', DictVectorizer(sparse=True)),
    ('scale', StandardScaler(with_mean=False)),
    ('model', ElasticNet(max_iter=20000, tol=1e-4))])
search = GridSearchCV(pipeline,
    {'model__alpha': [0.01, 0.1, 0.5, 1.0, 5.0],
     'model__l1_ratio': [0.1, 0.5, 0.9]},
    scoring='neg_mean_squared_error',
    cv=KFold(n_splits=5, shuffle=True, random_state=42), n_jobs=2)
search.fit(X_train, y_train)
best = search.best_estimator_
pred = best.predict(X_test)

baseline = DummyRegressor(strategy='mean')
baseline.fit(np.zeros((len(train), 1)), y_train)
baseline_pred = baseline.predict(np.zeros((len(test), 1)))
def scores(actual, prediction):
    return {'MAE_minutes': float(mean_absolute_error(actual, prediction)),
            'RMSE_minutes': float(np.sqrt(mean_squared_error(actual, prediction))),
            'R2': float(r2_score(actual, prediction))}

names = best.named_steps['encode'].get_feature_names_out()
coef = best.named_steps['model'].coef_
coef_table = pd.DataFrame({'feature': names, 'standardized_coefficient': coef,
    'coefficient_original_units': coef / best.named_steps['scale'].scale_})
coef_table['abs_standardized_coefficient'] = np.abs(coef)
coef_table.sort_values('abs_standardized_coefficient', ascending=False).to_csv(
    ROOT / 'coefficients.csv', index=False, encoding='utf-8-sig')
result = {'train_rows': len(train), 'test_rows': len(test),
    'best_parameters': search.best_params_,
    'CV_RMSE_minutes': float(np.sqrt(-search.best_score_)),
    'ElasticNet': scores(y_test, pred), 'mean_baseline': scores(y_test, baseline_pred),
    'feature_count': len(names), 'nonzero_coefficients': int(np.count_nonzero(coef)),
    'random_seed': 42}
(ROOT / 'model_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
pd.DataFrame(search.cv_results_).to_csv(ROOT / 'cross_validation.csv', index=False)
output = d.iloc[test][['show_id', 'title', 'duration_min']].copy()
output['predicted_minutes'] = pred
output['residual_minutes'] = y_test - pred
output.to_csv(ROOT / 'test_predictions.csv', index=False, encoding='utf-8-sig')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y_test, pred, s=10, alpha=0.35)
limits = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
axes[0].plot(limits, limits, 'r--')
axes[0].set(xlabel='Actual duration (minutes)', ylabel='Predicted duration (minutes)',
            title='Independent test set')
axes[1].scatter(pred, y_test-pred, s=10, alpha=0.35)
axes[1].axhline(0, color='r', linestyle='--')
axes[1].set(xlabel='Predicted duration (minutes)', ylabel='Actual - predicted (minutes)',
            title='Residuals')
fig.tight_layout()
fig.savefig(ROOT / 'model_diagnostics.png', dpi=150)
print(json.dumps(result, indent=2))
