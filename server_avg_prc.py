import pandas as pd
import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, average_precision_score
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

# サーバー側の評価関数（AUPRC用）
def get_evaluate_fn(model: LogisticRegression):
    def evaluate(server_round: int, parameters: NDArrays, config: dict):
        set_model_params(model, parameters)
        
        # AUPRCモデルの評価
        y_pred_proba = model.predict_proba(X_scaled)
        loss = log_loss(y, y_pred_proba)
        auprc = average_precision_score(y, y_pred_proba[:, 1])
        
        return loss, {"loss": loss, "auprc": auprc}
    return evaluate

# Flowerサーバーを起動してフェデレーテッドラーニングを実行する
if __name__ == "__main__":
    # ロジスティック回帰モデルを初期化（AUPRC用）
    model_auprc = LogisticRegression()
    set_initial_params(model_auprc)
    
    # FedAvg戦略の定義（AUPRC用）
    strategy_auprc = fl.server.strategy.FedAvg(
        min_available_clients=16,
        evaluate_fn=get_evaluate_fn(model_auprc)
    )
    
    # サーバーの起動
    fl.server.start_server(
        server_address="192.168.22.77:8080",
        strategy=strategy_auprc,
        config=fl.server.ServerConfig(num_rounds=10)
    )

    # 学習したモデルを保存
    filename_auprc_ela = 'finalized_model_auprc.sav'
    with open(filename_auprc_ela, 'wb') as f:
        pickle.dump(model_auprc, f)

    # 回帰係数をCSVに保存
    coef_auprc_df = pd.DataFrame(model_auprc.coef_, columns=feature_names)
    coef_auprc_df['intercept'] = model_auprc.intercept_
    coef_auprc_df.to_csv('AUPRC_model_coefficients_ela.csv', index=False)
    
    print("AUPRC最適化モデルが保存され、特徴量名と回帰係数がCSVファイルに出力されました。")

# 保存したモデルをロード
loaded_model_auprc = pickle.load(open(filename_auprc_ela, 'rb'))

# 新しいデータでモデルを評価
df = pd.read_csv('./GUSTO_sterberg.csv')
auprc_ave = []  # AUPRCのリスト

# 各地域（REGL値ごと）で評価
for i in range(1, 17):
    df_d = df[df['REGL'] == i]
    print(f"Region {i} shape: {df_d.shape}")
    
    # 特徴量とターゲットの準備
    X = df_d.drop(columns=['DAY30', 'REGL']).values
    y = df_d['DAY30'].values
    X_scaled = scaler.transform(X)
    
    # AUPRCモデルによる予測とAUPRCの計算
    y_pred_proba_auprc = loaded_model_auprc.predict_proba(X_scaled)
    auprc = average_precision_score(y, y_pred_proba_auprc[:, 1])
    auprc_ave.append(auprc)

# 結果をデータフレームとして保存
df_result = pd.DataFrame({
    'site': range(1, 17),
    'auprc': auprc_ave
})

# 結果の表示または保存
print(df_result)
df_result.to_csv('evaluation_results_AUPRC_ela.csv', index=False)  # モデルによって保存名を変える
