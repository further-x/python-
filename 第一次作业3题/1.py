import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from ISLP import load_data
    df = load_data('Carseats')
except ImportError:
    url = "https://raw.githubusercontent.com/intro-stat-learning/ISLP_labs/v2/ISLP/data/Carseats.csv"
    df = pd.read_csv(url)

model = smf.ols('Sales ~ Price + Income + Advertising + C(ShelveLoc)', data=df).fit()


print("=" * 60)
print("模型回归结果摘要：")
print("=" * 60)
print(model.summary())


X = model.model.data.orig_exog

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\n" + "=" * 60)
print("各变量的方差膨胀因子 (VIF)：")
print("=" * 60)
print(vif_data)