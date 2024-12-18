import flwr as fl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
import numpy as np

# データの読み込みと準備（クライアントごとに異なるデータを読み込む）
df = pd.read_csv('./GUSTO_sterberg.csv')
client_id = 2 # クライアントごとに異なるIDを設定する
df = df[df['REGL'] == client_id]

X = df.drop(columns=['DAY30', 'REGL']).values
y = df['DAY30'].values

# 入力特徴量の標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso（L1正則化）を用いたロジスティック回帰モデルの初期化
"""model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)"""
model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)
"""model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000)"""

# グリッドサーチのパラメータ設定（0.1から1.0まで0.1刻み）
param_grid = {'C': np.logspace(-3, 3, 7)}
"""param_grid = {'C': np.logspace(-3, 3, 7), 'l1_ratio': [0.1, 0.5, 0.9]}"""

# AUROCを最適化するためのスコアラーとグリッドサーチの設定
auroc_scorer = make_scorer(roc_auc_score)
grid_search_auroc = GridSearchCV(model, param_grid, scoring=auroc_scorer, cv=5, n_jobs=-1)
grid_search_auroc.fit(X_scaled, y)
best_model_auroc = grid_search_auroc.best_estimator_

# 最適なパラメータとスコアを表示
print(f"Best C value for AUROC: {grid_search_auroc.best_params_['C']}")
print(f"Best AUROC score: {grid_search_auroc.best_score_}")

# Flowerクライアントの定義
class FederatedClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [best_model_auroc.coef_, best_model_auroc.intercept_] if best_model_auroc.fit_intercept else [best_model_auroc.coef_]

    def set_parameters(self, parameters):
        best_model_auroc.coef_ = parameters[0]
        if best_model_auroc.fit_intercept:
            best_model_auroc.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        best_model_auroc.fit(X_scaled, y)
        return self.get_parameters({}), len(X_scaled), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred_proba = best_model_auroc.predict_proba(X_scaled)
        loss = log_loss(y, y_pred_proba)
        auroc = roc_auc_score(y, y_pred_proba[:, 1])
        return loss, len(X_scaled), {"loss": loss, "auroc": auroc}

# クライアントの起動
if __name__ == "__main__":
    client = FederatedClient().to_client()  # .to_client() メソッドで変換
    fl.client.start_client(server_address="192.168.22.77:8080", client=client)
