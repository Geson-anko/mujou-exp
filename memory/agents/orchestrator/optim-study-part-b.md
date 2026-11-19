# optim-study O0 part B（OGD・幅と μP・coord check）の作業記録

## 段階 1: 計画

- fix: grad_stats の `torch.linalg.eigh` が CUDA の lazy linalg のロードで node スレッド同士が衝突する（mini-00 の smoke）。gram は k×k（k ≤ 16）なので CPU で解く。OGD も同じ道を通るので先に直す
- OGD: `exp/lib/ogd.py` の `OGD(base=SGD|RMSProp|AdamW, n, every)`。base の更新は mujou と同じ演算順で書き直す（記憶が空なら bit 一致）。射影は flat なベクトルで、gram と係数は CPU の float64
- 保持は GradStats と同じ規則（ノード自身の elapsed の和 t、t + elapsed/2 > kP の最初の歩、最初の歩で 1 回）
- `optim.ogd_n`（0 で無効）・`optim.ogd_every`（既定 1.0、ogd_n = 0 なら既定のまま）。sgd・rmsprop・adamw だけ
- 幅: `model.heads` を `int | None`（None → width / 64、post_init で解決）。`model.scale_decoder`、`model.param`（sp | mup）
- μP の初期化: 基準の幅 256 に錨を置き、隠れ層は分散 ∝ 1/fan_in、出力は ∝ 1/fan_in²（= 幅 256 の std × 1/k）。いまの SP の初期化に対する倍率は、ViT のブロック（std 0.02 固定）×k^-1/2、デコーダの Conv（torch 既定 ∝ fan_in^-1/2）×1、head・readout ×k^-1/2、入力 ×1
- 学習率の倍率: `exp/lib/optim.py` の `ScaledLR(inner, scale)`。倍率ごとに inner の tau を tau/m にして束ねる。全部 1 なら inner.bind そのもの（bit 一致）
- coord_check: Online を task=bench で回し、最初のスタックを probe に各ノードの出力の RMS と初期値からの変化

## 段階 3〜4: 計画外の判断と自己レビュー

- `heads` を post_init で解決すると、OmegaConf が schema の既定値を作る時点で 4 に固まり、`model.width=512` でも 4 のままだった。property `head_count` で読む形に変えた
- `ScaledLR` は OGD を包めない（tau が無い）。OGD の射影はノード全体のベクトルで行うので、群に分けると意味が変わる。`model.param=mup` と `optim.ogd_n > 0` は RunConf で拒む
- 手元の CPU の coord check（合成のスタック）: μP の隠れ層の delta の傾き ≈ 0、SP は正。出力の側は μP で −0.4〜−0.5。probe と学習の入力が無相関なためと読んだ。実データの GPU で確かめる
- 自己レビュー: ノードをまたぐ backward / autograd の追加なし（grep）。OGD・GradStats・ScaledLR の状態はノードの BoundOptimizer の中だけ
