# 基礎 F1〜F3 の学習の側: 回す前に決めたこと・実装・試し

[plans/foundation.md](../../plans/foundation.md) の 3 節（原理）の学習を、`jepa_iid.py`（BP + AdamW、i.i.d. なミニバッチ）の上に実装した。腕は CLI の `--arm` で切り替える。結果の主の記録は [README.md](README.md) にあり、この文書は学習の側の決定・実装・計算機の試しを書く。

## 回す前に決めたこと（2026-11-27）

### $s$ の上端は 128 のまま

学習の動画の長さを fleet のデータ（`ad6e56b5995e`）の `index.json` で調べた。Zurich を除く 9 本のフレームの数は次のとおりである。

| 動画 | フレーム |
| -- | -- |
| Amsterdam | 49126 |
| Bangkok | 105000 |
| Chiang_Mai | 40752 |
| Istanbul | 40800 |
| Kuala_Lampur | 43700 |
| Singapore | 58004 |
| Stockholm | 39895 |
| Venice | 65999 |
| Wildlife | 36074 |

$s = 128$ の stack は $7 \cdot 128 + 1 = 897$ フレーム（約 90 秒）にわたる。9 本のどれにも入るので、「学習の動画の 90% 以上で引ける」条件を満たす（9 本中 9 本）。上端は下げずに $s_{\max} = 128$ とした。始点のうち除かれる割合は最も短い Wildlife で $896 / 36074 = 2.5\%$、9 本の合計で約 1.7%（$9 \cdot 896 / 479350$）である。読み出しは 1 フレームずつ索引で引くので、$s$ によって速さは変わらなかった（下の試しで、Sσ の 1 step は S1 と同じ程度）。

### 実装の取り決め

計画にない細部は次のように決めた。

- **マスクの縦横比の引き方**: V-JEPA は縦横比を batch ごとに 1 回引き、全ての stack と全てのブロックで共有する。ここでは stack ごとに 1 回引き、その stack のブロックで共有した。1 つの stack から見た分布は V-JEPA と同じになる。違うのは batch の中の stack どうしの相関だけである
- **短距離と長距離の割り当て**: V-JEPA は 1 つの clip に両方を当て、重みを等しくする。ここでは batch の前半を短距離、残りを長距離にした（B = 128 なら 64 と 64）。期待値は同じで、計算は 1 回分で済む
- **整数の一様分布**: ブロックの長さ・始まりの時刻・上端・左端は、float64 の一様乱数 $u$ から $\lfloor u n \rfloor$ で作る
- **予測器の $s$**: Sσ だけが $\psi(\log_2 s)$ を予測器の位置に足す。T0 と S1 は足さない（$s = 1$ の定数になるため）
- **attention**: `scaled_dot_product_attention` で計算する。文脈の側は見えるトークンだけを batch の最大の数まで詰め、詰め物は鍵から外す。`models.Block` と同じ計算であることはテストで確かめた
- **検証の組**: dyn/copy の検証の動画（`split_videos` の seed 4101 で選ばれる `val`。JEPA の学習の動画でもあり、V18c からの分け方のまま）を使う。Zurich と dyn/copy の記録の動画（`rec`）は使わない。補完の損失には腕の $s$ の分布とマスクで 512 個の stack を `VAL_SEED` = 4301 から引く。崩壊の指標には $s = 1$ の 1024 個の stack を動画全体に等間隔に置く。held-in には学習の動画の 4 × 128 個の stack を `HELDIN_SEED` = 4303 から引く。どれも run の seed によらないので、同じ腕の run どうしで比べられる

## 実装

| ファイル | 中身 |
| -- | -- |
| `exp/vision/jepa.py` | `sincos_3d`（時空間の位置。$e_P(p) + \phi(u_j)$、$u_j = j - 3/2$）、`TubeletEmbed`（2 フレーム × 8 × 8 画素の箱。`PatchEmbed` の子で、muP では input の役）、`TokenEncoder.vit` |
| `exp/vision/iid.py` | `Units.draw_spaced`（間隔 $s$ の stack。1 つの訓練の動画の中に収め、最後のフレームを、その stack が入る所から一様に引く） |
| `exp/vision/fnd_jepa.py` | 腕（`ARMS`: T0・S1・Ss）、マスク（`draw_blocks`・`draw_masks`）、間隔（`draw_spacings`）、`build_fnd`、`attend`・`encode`・`complete`・`fnd_terms`、`time_r2`、checkpoint の符号（`load_fnd`・`FndCode`・`z_fn`） |
| `exp/vision/fnd_val.py` | 検証の動画の数（`val_set`・`val_rows`） |
| `exp/vision/jepa_iid.py` | `--arm {T0,S1,Ss}`・`--s-max`（既定 128）と `train_fnd`。`--arm` を付けなければ V18c までの挙動のまま（`--mask-ratio` は `--arm` の無いときだけ必須） |

`--arm` の run は $d = 16$、$|\Omega| = 4 \times 8 \times 8 = 256$、$D = 4096$ の正方の $W$ を、固定の値として持つ。CLI の `--token-dim`・`--proj`・`--sigreg-on`・`--mask-ratio` は読まない。`--patch 8` は渡す（既定の設定はパッチ 16）。

```bash
python -m exp.vision.jepa_iid --patch 8 --arm S1 --lam 0.8 --lr 1e-4 --steps 6000 --out <run>
```

- `train.jsonl`（`--log-every` ごと）: `loss`・`pred`・`sigreg`・`grad_norm`・`t_var`・`z_var`・`time_r2`（batch のトークン）・`hidden`（隠したトークンの割合）・`pred_short`・`pred_long`・`lr`・`sec`・`max_mem`
- `val.jsonl`（checkpoint の step ごと。step 0 を含む）: `val_pred`・`val_rel`（予測器の $R_{\mathrm{clim}}$）・`val_rel_short`・`val_rel_long`・`sigreg_heldin`（K1）・`rank`（K2）・`token_r2`（K3）・`time_r2`・`participation`・`sigreg_val`・`sigreg_gauss`・`t_var`・`z_var`
- `ckpt/<t>.pt`: `encoder`・`proj`（$R$）・`predictor`・`mix`（$W$）。`ckpt/00000.0.pt` が学習前（step 0）である
- **符号の取り出し**: `exp.vision.fnd_jepa:z_fn` は、`fnd_dyn` の `--places NAME exp.vision.fnd_jepa:z_fn <run>/ckpt/<t>.pt` で使える。stack（u8、`(B, 8, 3, 64, 64)`）から $z$（`(B, 4096)`）への写像を返す。`FndCode(mods, heads)` は `mu`（$z$）と `tokens`（$t$、`(B, 256, 16)`）を持つ。どちらも的の側（全てのトークンを見せた符号）である

### テストのオラクル

| テスト | 中身 |
| -- | -- |
| 箱（`TestTubelets`） | 時刻 1 つの箱は、同じ重みの `PatchEmbed` とビットまで一致する。トークンは自分の箱の線形写像に一致し、1 画素は自分の箱にしか届かない。位置の表に $\psi$ を足すと `Predictor.st_positions` にビットまで一致する。時間の反転で、$\phi$ は sin だけ符号が変わる |
| マスクの族（`TestBlocks`） | 辺の長さは V-JEPA の規則で手計算した値（8 × 8 で短距離 (3,3)・(4,2)、長距離 (6,8)・(8,5)）と一致する。辺の組の分布は、縦横比の一様分布から出る閉じた式とカイ二乗で一致する。位置は入る所で一様、長さと始まりの時刻も一様である。1 ブロックが時刻 $j$ を覆う確率は閉じた式と一致し、反転で対称になる。SHORT と LONG の隠す率は、時間の反転と縦横の鏡映で不変である。T0 は時刻 1 つの画像の multi-block を全時間に伸ばした管と、同じ乱数でビットまで一致する（V-JEPA の multi-block）。時刻が 1 つなら、時間を引く族と引かない族がビットまで一致する（I-JEPA の multi-block の形）。全部を隠すマスクは出ない。隠す数は固定されない |
| 間隔（`TestSpacings`・`test_iid.TestSpaced`） | T0・S1 は $s = 1$。Sσ は `draw_lags(1, s_max)`（丸めた対数一様。法則は `TestDrawLags`）とビットまで一致する。$s = 1$ の stack は `Units.draw` とビットまで一致する。フレームは $s$ おきに並び、1 つの訓練の動画に収まる。最後のフレームは、入る所で一様である。$s < 1$ と入らない $s$ は拒む |
| attention・損失（`TestModules`・`TestLoss`） | `attend` は `Block` と一致する。詰め物は鍵にならない（その stack だけで計算した値と一致する）。各 stack の誤差は、その stack だけを詰め物なしで手計算した値と一致する。`loss` $= (1-\lambda)$`pred` $+ \lambda$`sigreg` が成り立つ。隠した箱の画素を変えても、予測はビットまで変わらない（P4）。隠した画素にも勾配が届く（stop-gradient が無い）。見える時刻の画素を変えると、隠した時刻の的が動く（P5。漏れが開いていることの記録）。activation checkpoint をかけても勾配は一致する。$W$ は seed 61 の正方の行列で、どちらも optimizer に入らない。埋め込みは LeCun の打ち切り正規分布（2σ）である。他のブロックと予測器は V18c と同じ seed の値になる |
| `time_r2`（`TestTimeR2`） | 時間に一定のトークンで 1、時刻どうしが独立で分散が等しいトークンで $1/T$ になる。定数のずれと時刻の並べ替えに左右されない |
| SIGReg の罰（`TestSigRegPenalties`） | 正方の $W$（4096）と方向 1024 本で $N = 128$ のとき、K3（$q = 16$）の期待値の超過は等方より $0.241\,N(1/q - 1/D) = 1.92$ の 10% 以内（実際は 1.84）、時間に一定のトークン（$q = 1024$）は 0.0226 の 12% 以内（実際は 0.0205）。`sigreg` の実装を 120 回引いた平均は、3 つの場合とも期待値の 5 SE 以内に入る |
| 検証の数（`test_fnd_val.py`） | 動画は dyn/copy の `val` で、Zurich ではない。stack は腕の間隔とマスクを `VAL_SEED` から引く。`val_rel`・種類ごとの `val_rel`・`val_pred`・`rank`・`time_r2`・`token_r2` は、手計算した値と一致する |
| CLI（`test_jepa_iid.TestArms`） | 腕ごとの設定・行・検証の行が有限で、checkpoint は step 0 と最後に残る。同じ seed なら同じ run になる。step 0 は `build_fnd` そのもので、$R$ と $W$ も残る |

## 計算機の試し（2026-11-27、snapshot `75c782986e85`）

場所は `/nas/geson/mujou-exp/runs/fnd-train-trial/`（`scripts/` に job と試しのスクリプト、`logs/` に各 run のログと nvidia-smi を 2 秒ごとに記録したもの）。設定は λ 0.8、lr 1e-4、warmup 20、60 step、batch 128。1 step の時間は `train.jsonl` の step 10 から 60 までの `sec` の差を 50 で割ったものである（step 0 の検証の時間を含まない）。

### 4090 に 1 本（checkpoint なし）

| 腕 | ホスト | 1 step（秒） | `max_mem`（GiB、PyTorch の割り当て） | nvidia-smi の最大（MiB） |
| -- | -- | -- | -- | -- |
| T0 | std-00 | 0.488 | 16.96 | 22815 |
| S1 | std-01 | 0.583 | 18.51 | 23487 |
| S1 | std-03 | 0.546 | 18.51 | 23323 |
| Sσ | std-02 | 0.546 | 18.54 | 23355 |

T0 はマスクが管だけで、隠す割合が大きい（`hidden` 0.72。S1・Sσ は 0.56）。そのため文脈の側のトークンが少なく、速い。検証の動画の数の計算（`val_rows`）は 1 回 4〜6 秒である。

### 4090 に複数本（run ごとに別のプロセス）

- **checkpoint なしの 2 本**: 同時に起動すると、2 本とも最初の step で CUDA の OOM になった（std-03。1 本が 9.4 GiB と 14.0 GiB を持った時点で、空きが 73 MiB）。1 本で 23 GB 近くを使うので、4090 には 1 本しか載らない
- **`--checkpoint-blocks` の S1**（std-00 と std-03 で同じ結果）:

| 本数 | 1 本の 1 step（秒） | 合計で 1 step あたり（秒） | nvidia-smi の最大（MiB、std-03） |
| -- | -- | -- | -- |
| 1 | 0.666 | 0.666 | 5627 |
| 2 | 1.446 | 0.723 | 11217 |
| 3 | 2.17 | 0.723 | 16806 |

1 本でも GPU の使用率は 100% だった。本数を増やしても合計の速さは上がらない。checkpoint なしの 1 本（0.49〜0.58 秒）が最も速い。そこで**学習の run は 4090 に 1 本、checkpoint なし**とする。2〜3 本を同居させても illegal memory access は起きなかった（std-00・std-03 で 6 本）。

### 小さい GPU（`--checkpoint-blocks`、S1、30 step）

| GPU | 1 step（秒） | `max_mem`（GiB） |
| -- | -- | -- |
| 4060（std-01） | 3.42 | 4.66 |
| 5070 Ti（mini-00） | 1.02 | 4.66 |

checkpoint なしでは 18.5 GiB が要り、4060（8 GB）にも 5070 Ti（16 GB）にも載らない。

### 見込みの時間

| run | 4090（1 本、checkpoint なし） | 5070 Ti | 4060 |
| -- | -- | -- | -- |
| 短い（6000 step） | 49〜58 分 | 約 1.7 時間 | 約 5.7 時間 |
| 長い（60000 step） | 8.1〜9.7 時間 | 約 17 時間 | 約 57 時間 |

検証の数の計算は、短い run で 5 回、長い run で 9 回、1 回 4〜6 秒なので、時間にはほとんど効かない。長い run は計画の見込み（6〜8 時間）より 1〜2 時間長い。

### 再現性の注意

同じ seed の S1 を 2 台（std-01 と std-03）で回すと、`loss` は表示した桁まで一致した。一方 `grad_norm` は相対で $10^{-8}$ ほどずれた。CUDA の memory-efficient attention の backward は決定的でない（PyTorch が警告を出す。`configure_determinism` は CUDA では警告にとどめる）。CPU では同じ seed でビットまで同じになる（テスト）。
