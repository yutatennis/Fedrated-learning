import pandas as pd
import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from flwr.common import NDArrays

# データの読み込みと準備
df = pd.read_csv('./GUSTO_sterberg.csv')
df = df[df['REGL'] == 1]
X = df.drop(columns=['DAY30', 'REGL']).values
y = df['DAY30'].values

# 入力特徴量の標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 特徴量名の取得
feature_names = df.drop(columns=['DAY30', 'REGL']).columns

# モデルのパラメータを設定する関数
def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

# モデルの初期パラメータを設定する関数
def set_initial_params(model: LogisticRegression):
    n_classes = 2
    n_features = X_scaled.shape[1]
    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

# サーバー側の評価関数（AUROC用）
def get_evaluate_fn(model: LogisticRegression):
    def evaluate(server_round: int, parameters: NDArrays, config: dict):
        set_model_params(model, parameters)
        
        # AUROCモデルの評価
        y_pred_proba = model.predict_proba(X_scaled)
        loss = log_loss(y, y_pred_proba)
        auroc = roc_auc_score(y, y_pred_proba[:, 1])
        
        return loss, {"loss": loss, "auroc": auroc}
    return evaluate

# Flowerサーバーを起動してフェデレーテッドラーニングを実行する
if __name__ == "__main__":
    # ロジスティック回帰モデルを初期化（AUROC用）
    model_auroc = LogisticRegression()
    set_initial_params(model_auroc)
    
    # FedAvg戦略の定義（AUROC用）
    strategy_auroc = fl.server.strategy.FedAvg(
        min_available_clients=16,
        evaluate_fn=get_evaluate_fn(model_auroc)
    )
    
    # サーバーの起動
    fl.server.start_server(
        server_address="192.168.22.77:8080",
        strategy=strategy_auroc,
        config=fl.server.ServerConfig(num_rounds=10)
    )

    # 学習したモデルを保存
    filename_auroc_L2 = 'finalized_model_auroc.sav'
    with open(filename_auroc_L2, 'wb') as f:
        pickle.dump(model_auroc, f)

    # 回帰係数をCSVに保存
    coef_auroc_df = pd.DataFrame(model_auroc.coef_, columns=feature_names)
    coef_auroc_df['intercept'] = model_auroc.intercept_
    coef_auroc_df.to_csv('AUROC_model_coefficients_L2.csv', index=False)
    
    print("AUROC最適化モデルが保存され、特徴量名と回帰係数がCSVファイルに出力されました。")

# 保存したモデルをロード
loaded_model_auroc = pickle.load(open(filename_auroc_L2, 'rb'))

# 新しいデータでモデルを評価
df = pd.read_csv('./GUSTO_sterberg.csv')
auroc_ave = []  # AUROCのリスト

# 各地域（REGL値ごと）で評価
for i in range(1, 17):
    df_d = df[df['REGL'] == i]
    print(f"Region {i} shape: {df_d.shape}")
    
    # 特徴量とターゲットの準備
    X = df_d.drop(columns=['DAY30', 'REGL']).values
    y = df_d['DAY30'].values
    X_scaled = scaler.transform(X)
    
    # AUROCモデルによる予測とAUROCの計算
    y_pred_proba_auroc = loaded_model_auroc.predict_proba(X_scaled)
    auroc = roc_auc_score(y, y_pred_proba_auroc[:, 1])
    auroc_ave.append(auroc)

# 結果をデータフレームとして保存
df_result = pd.DataFrame({
    'site': range(1, 17),
    'auroc': auroc_ave
})

# 結果の表示または保存
print(df_result)
df_result.to_csv('evaluation_results_AUROC_L2.csv', index=False) #モデルによって保存名を変える
