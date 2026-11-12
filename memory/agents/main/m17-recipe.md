# M17: 原典のレシピと、私たちの設定との違い（2026-11-12）

M17 の試しの run（results/m17.md）が no-go になったのを受けて調べた。対象は I-JEPA（arXiv 2301.08243）・V-JEPA（2404.08471）・V-JEPA 2（2506.09985）・LeJEPA（2511.08544、v3）と、LeJEPA 公式リポジトリの `MINIMAL.md`。原文は arXiv の HTML 版から読んだ。節・表の番号はその版のもの。コードは変えていない。

## 0. 要点

1. **ハイパラは不具合の主な原因ではなさそう**。warmup は短くない（300 秒 ≈ 1.2 万 step で、V-JEPA と同じ step 数。600 秒の run なら 50%）。lr 1e-4 も、原典の範囲（2e-4〜6.25e-4、LeJEPA 2e-3〜5e-3）と比べて高すぎない。原典との大きな違いは **weight decay 0**（原典は 0.04〜0.4）、**減衰のない一定の lr**、**B=32 で i.i.d. でないバッチ**（原典は 2048〜3072 の i.i.d.）、**浅い予測器**（2 層・幅 256。原典は 6〜12 層・幅 384）。どれも直す価値はあるが、「学ぶほど学習前より悪くなる」の主因とは考えにくい
2. **主因の候補は目的の式にある近道**。(a) EMA も sg もないトークン予測では、target のトークンを「位置の符号 + 場面の色」だけにすれば予測誤差を 0 に近づけられる。SIGReg は pool した $z$ にしかかからず、1 つのクリップの中でトークンがつぶれることを止めない（`both` もトークンの項を $1/T$ に縮めている）。(b) 9 本の動画を流す 32 本のストリームでは、SIGReg は「どの動画・どの場面か」（色・明るさ）だけで満たせる。LeJEPA は view ごとの色の拡張でこの近道を消しているが、私たちは拡張を使っていない。観察（予測器が場面の平均色を出す、`pred` が 1.06→0.06 に下がるのに何も学ばない、SIGReg を強めると学ぶ 9 本の見分けだけが進む）は、この 2 つの近道でそのまま説明できる
3. **masked-token JEPA を EMA なし・SIGReg だけで学べる、という公表された証拠は見つからなかった**。LeJEPA が「予測器も teacher-student も要らない」と示したのは（§6.1、Table 4）、拡張した view の pool した埋め込みを互いに当てる不変性の形だけ。トークンのマスク予測で EMA を外した報告はない。近いのは、凍った教師を使う SALT（2509.24317）と、I-JEPA に VICReg を足した C-JEPA（2410.19560）
4. **ユーザーの方針「時空間の補完」は原典にも裏づけがある**。V-JEPA Table 4 では、時空間の multi-block（K400 72.9 / SSv2 67.4 / IN1K 72.8）が、過去のフレームだけを context にする causal multi-block[6]（61.3 / 49.8 / 66.9）より大きく上。次フレーム（未来）だけを当てる課題より、時空間の補完が良い特徴量を作る
5. **提案: 時空間の LeJEPA（ST-LeJEPA）**。同じクリップの部分 view（空間のマスク・過去だけ・空間の切り出し）の pool した埋め込みを、全体の view の埋め込み $\mu$ に一致させる（LeJEPA Eq. 6〜9）。SIGReg は view ごとの projector の出力にかける。部分しか見ない view の $z$ が、見えない領域と未来を含むクリップ全体の $z$ と一致する必要があるので、これが $z$ の水準の時空間の補完になる。トークンの予測は補助の項にし、target は LayerNorm してから使う。色の近道は、view ごとの色の拡張で塞ぐ（§6）
6. **先に、正しさを確かめる基準の課題を回す**。LeJEPA の minimal（ViT-S/8、Imagenette 128 px、`lamb=0.02 V=4 proj_dim=16 lr=2e-3 bs=256 epochs=800` で線形 probe 90.7%、200 分）を、私たちの ViT・SIGReg・mujou の AdamW に置き換えながら再現する。4090 1 台で 1 時間以内に収まる。学習前の ViT の probe との差を見て、実装に不具合がないことを確かめる（§4）
7. **評価にも直すところがある**。val の SIGReg（8〜12）を学ぶ側（2.1）と比べるのは公平でない。val のバッチは、時間順に並べた Zurich の窓を 32 個ずつ切ったもので（`eval_clips` が slot を `sort()` してから split に分ける）、1 本の動画の数分（Zurich を約 1.4 時間とみて 3〜4 分）から来る。学ぶ側のバッチは 9 本に散らばる。val の SIGReg が高いのは、汎化できていないことを必ずしも意味しない

## 1. 原典のレシピ（小さいモデルの数字があればそれを優先）

| 項目 | I-JEPA（App. A・C） | V-JEPA（App. C Table 8） | V-JEPA 2（App. A Table 9・10） | LeJEPA 論文（§6.1 Experiment Details 1） | LeJEPA `MINIMAL.md`（ViT-S/8、Imagenette） | M17 の試し（BP） |
| -- | -- | -- | -- | -- | -- | -- |
| モデル | ViT-B/16〜H/14 | ViT-L/H（E.4 の mask の ablation は ViT-B/16、K400） | ViT-L〜g、予測器は ViT-s（Table 12） | timm の各種（ImageNet-10 は 20M 未満の約 50 種、Fig. 9） | `vit_small_patch8_224`、`img_size=128` | 幅 256・深さ 8・4 heads・patch 16 |
| 最適化 | AdamW | AdamW | AdamW | AdamW | AdamW | mujou の `pc.optim.AdamW`（時間の形） |
| batch | 2048 | 3072 | 3072 | ≥128 を推奨（Table 1c: 128→72.20、256→74.15、512→74.72、1024→74.07） | 256 × V=4 view | 32（ストリーム、i.i.d. でない） |
| lr | 1e-4 から 1e-3 へ線形、その後 cosine で 1e-6 | 2e-4 から 6.25e-4 へ、cosine で 1e-6 | 1e-4 から 5.25e-4 へ、一定、cooldown で 1e-6（Table 9）。短い版は 2e-4→6.25e-4→1e-6（Table 10） | {5e-3, 5e-4}、線形 warmup + cosine | 2e-3、cosine で `eta_min=1e-3`（絶対値なので最後も 1e-3） | 1e-4/frame（約 7.5e-5/step、下の注）、warmup の後は一定 |
| warmup | 15 epoch / 300（5%） | 12000 / 90000 iter（13%。予定表は 112500 で計算、§C） | 12000 step、その後は一定 | 長さの記載なし | 1 epoch / 800（36 step、0.13%）、`start_factor=0.01` | 300 秒 ≈ 12000 step（600 秒の run の 50%、1800 秒の 17%） |
| 総 step | ViT-H 300 epoch ≈ 18.8 万 | 90000 | 25.2 万 + cooldown 12000 | ImageNet-1k 100 epoch、IN-100 400 epoch、Galaxy10 400 epoch | 800 epoch ≈ 2.9 万 | 600 秒 ≈ 2.4 万、1800 秒 ≈ 7.2 万 |
| weight decay | 0.04→0.4 を線形（Table 13: 固定の 0.05 は linear-eval で 76.4、0.04→0.4 は 77.8） | 0.04→0.4 | 0.04 固定 | {1e-1, 1e-2, 1e-5}、予定表なし | 5e-2（probe は 1e-7） | **0** |
| betas | 論文に記載なし（公式コードは torch の既定 (0.9, 0.999) と理解。未確認） | 同左 | 同左 | 同左 | torch の既定 (0.9, 0.999) | $\tau_m$ 0.3 s・$\tau_v$ 30 s、step 25 ms で β1 ≈ 0.920、β2 ≈ 0.99917 |
| EMA | 0.996→1.0 を線形 | 0.998→1.0 | 0.99925 固定（短い版は 0.999→1.0） | なし（ViT は任意で SWA。§6.1） | なし | なし（規則） |
| 予測器 | 幅 384、ViT-B なら深さ 6、L/H は 12、G は 16（App. A）。Table 12: 6 層 64.0、12 層 66.9（ViT-L、IN-1%）。Table 14: 幅 384 は 70.7、1024 は 68.4 | 12 block・幅 384 | ViT-s（幅 384・深さ 12・12 heads） | 使わない（§6.1、Table 4） | なし | 2 block・幅 256（エンコーダと同じ） |
| マスク・view | target 4 ブロック（scale 0.15〜0.2、aspect 0.75〜1.5）、context 1 ブロック（0.85〜1.0、重なりを除く） | 短い範囲 8 × 0.15、長い範囲 2 × 0.7、aspect 0.75〜1.5、時間方向は全部。平均で約 90% を隠す（E.4） | 空間 [0.15, 0.7]、時間 [1, 1]、aspect 0.75〜1.5 | global 2 枚（224）＋ local（96）、計 V=8。マスクはなし | 4 view、`RandomResizedCrop(128, 0.08–1.0)`＋色の拡張 | V-JEPA と同じ 2 種類の tube。拡張なし |
| clip | — | 16 フレーム、stride 4（30 fps で約 2 秒、App. B）、2×16×16 の tubelet | 16 フレーム、4 fps（4 秒）。cooldown で 64 フレーム | — | — | 16 フレーム、間隔 3（0.1 s、1.5 秒）、$\tau=2$ |
| 拡張 | なし | random resize crop (0.3, 1.0)、aspect (0.75, 1.35)、hflip | 同左 | DINO の multi-crop と色の拡張（README: ColorJitter 0.4/0.4/0.2/0.1 p=0.8、grayscale 0.2、blur 0.5、solarize 0.2） | ColorJitter(0.8,0.8,0.8,0.2) p=0.8、grayscale 0.2、blur、solarize 0.2、hflip | なし |
| 損失 | L2 | L1（Eq. 2） | L1（Eq. 1） | L2（Eq. 6〜8） | L2 | L2（`pred_loss` は軸） |

注（私たちの step）: `time.learn_interval` は 25 ms、`time.frame` は 1/30 s なので、BP は約 1.33 step/frame で学ぶ（3 frame に 4 回。同じ frame で 2 回学ぶこともある）。`make_optimizer` は $\tau = \mathrm{frame}/\mathrm{lr}$ なので、1 step の $\eta = 0.025 / (0.0333/10^{-4}) = 7.5\times10^{-5}$。600 秒の run は約 2.4 万 step、B=32 で約 77 万 sample になる。ただし隣り合う積みは 16 枚のうち 15 枚を共有し、ストリームは 1 step で 0.075 s しか進まない。独立な sample の数は、名目の数よりずっと少ない。

注（warmup の実装）: `exp/lib/optim.py::Warmup` は時間そのものを $r(t)$ で伸縮させる。ramp の間は、Adam の moment の時定数も $1/r(t)$ 倍に伸びる。torch の `LinearLR` は lr だけを ramp し、moment は普通に動く。初めのうちは moment がほとんど更新されない点が原典と違う。影響は小さいとみるが、基準の課題（§4）で torch の形と比べられる。

注（バッチの大きさと lr）: V-JEPA の 6.25e-4 @ 3072 を B=32 に換算すると、線形則で 6.5e-6、平方根則で 6.4e-5。私たちの 7.5e-5/step は平方根則の上側にあたる。ただし LeJEPA の minimal は B=256 で 2e-3 と高く、LeJEPA は lr に頑健だと主張している（§6.1）。診断では 3e-4 が 1e-4 より悪かった。1e-5〜3e-5 はまだ試していない。

## 2. LeJEPA の目的の正確な形（§5.1）

- **view**: DINO の形で、global $V_g$ 枚と local $V_l$ 枚、計 $V$ 枚（推奨は $V_g=2$、$V_l=8$。§6.1）。ResNet では global = 全部の view にする。マスクは使わない
- **予測するもの**: トークンではなく、view ごとの埋め込み $z_{n,v}$（backbone → projector の出力）。中心 $\mu_n = \frac{1}{V_g}\sum_{v \le V_g} z_{n,v}$（Eq. 7）に、すべての view を寄せる: $\mathcal{L}_{\rm pred} = \frac1V\sum_{v'}\lVert\mu_n - z_{n,v'}\rVert_2^2$（Eq. 8。global 自身も入る）。予測器・EMA・stop-gradient はない（§6.1 は「予測器も register token も使わない」ことを推奨）。$\mu$ にも勾配が流れる（コード `centers = g_emb...mean(0)` に detach がない）
- **全体**: $\mathcal{L} = \frac{\lambda}{V}\sum_v \mathrm{SIGReg}(\{z_{n,v}\}_{n=1}^B) + \frac{1-\lambda}{B}\sum_n \mathcal{L}_{\rm pred}$（Eq. 9）。$\lambda = 0.05$ を推奨（§6.1、Fig. 8）。minimal は 0.02。Table 7（ResNet-50、IN-100、400 epoch、V=2）: λ 0.001→81.41、0.005→82.73、0.01→83.49、0.02→82.99、0.025→82.23
- **SIGReg**: Epps-Pulley を 1 次元への射影ごとに計算し、平均する。方向は step ごとに引き直す（§4.3。16 方向を引き直すほうが、固定した数千方向より良い）。推奨は 1024 slices（Table 1a: 512 で 74.21、2048 で 74.80）。コードと minimal は 256。積分は [−5, 5] の 17 点の台形則（minimal は対称性を使って [0, 3] の 17 点）、窓は $e^{-t^2/2}$。統計量は $N$ 倍され、minibatch の偏りは $O(1/N)$ で、batch 16 でも小さい（§5.1、Theorem 6）。**view ごとに**、batch 方向にかける（`mean(SIGReg(emb) for emb in a_emb)`）。かける先は projector の出力（minimal の `proj`、512→2048→2048→16、BN つき MLP）。probe は backbone の出力（`emb`、512 次元）で取る
- **projector**: Table 4（IN-100、400 epoch）では 3 層が最良。ViT-S/8 で、予測器なし・SWA なしは 1/2/3 層で 76.59/80.77/81.07、SWA ありは 79.96/83.63/84.12。予測器を足すと同等か悪い（77.58/79.41/81.91）。Table 1d: proj. dim 64 が 75.29〜75.65 で最良の側（ViT-L）
- **損失と精度の相関**: 学習の損失と線形 probe の精度の Spearman 相関は約 85%。$\mathrm{loss}/\lambda^{0.4}$ にすると 99% まで上がる（§6.2、Eq. 10、Fig. 11）。SIGReg と予測の損失の 2 次元平面に前線がある（Fig. 10）。ユーザーの「補完の誤差を主な指標に」と合う。ただしこれは λ を固定した損失の比較で、両方の項を一緒に見る

## 3. 小さい基準の課題（実装が正しいかを確かめる用）

| 出典 | 設定 | 線形 probe（frozen） |
| -- | -- | -- |
| `MINIMAL.md`（2025-11-20） | ViT-S/8、Imagenette 160px 版を 128 に切る、V=4、λ=0.02、proj_dim 16、lr 2e-3、wd 5e-2、bs 256、800 epoch、bf16 | **90.7%（200 分、GPU は記載なし）**。「1 台で約 2 時間」とも書いてある |
| 論文 Table 5（in-domain、ランダムな初期値から） | inet10（13000 枚）: convnextv2_nano 90.81、resnet18 92.36、resnet34 92.80、resnext26ts 92.82、swin_tiny 92.87。cifar10（50000）: 92.22〜93.16。cifar100: 65.08〜70.44。galaxy10（11008）: 73.78〜77.29 | epoch は Galaxy10 だけ 400 と書いてある（§6.3） |
| 論文 Fig. 9 | ImageNet-10、timm の 20M 未満の約 50 種 | 91.5〜95% |
| 論文 Table 3 | Galaxy10、frozen、全ラベル: ResNet-34 78.17、ConvNeXt-V2 Nano 76.52。DINOv3 ViT-S/16 は 71.38 | — |

- **学習前のモデルの probe は、論文にも minimal にも載っていない**。自分たちで測る必要がある。Imagenette は 10 クラスで、偶然の正解率は 10%。学習前の ViT の線形 probe は経験的に 30〜45% 程度と見込む（出典なしの見積もり。最初の run で確かめる）
- **4090 1 台・1 時間以内の案**: Imagenette、128 px、minimal の設定のまま。ただし backbone を私たちの ViT（幅 256・深さ 8・patch 16 → 64 トークン）に替える。ViT-S/8（256 トークン・幅 384・深さ 12）より 1 step の計算は 1 桁ほど軽い見込みで、800 epoch（約 2.9 万 step）でも 1 時間に収まると見る。期待値は ViT-S/8 の 90.7% より下（patch 16 で解像度が粗い）。合格の目安は「学習前 + 30 ポイント以上、online probe が単調に上がる」。数字を固定するのは、最初の run を見てから
- 段階: (S1) minimal をそのまま、timm の ViT-S/8 で 100 epoch（環境の確認） → (S2) backbone・SIGReg（`exp.lib.sigreg`）・最適化（`pc.optim.AdamW` + `Warmup`）を 1 つずつ私たちのものに替える → (S3) view を I-JEPA のマスク予測に替え、EMA なし・SIGReg だけにする（§5 の未解決の問いに、ラベルのある画像で答える）。依存は `uv add`（`pip` は使わない）で、`tools/` の別プロジェクトに置くのがよい

## 4. EMA なしの masked-token JEPA について分かっていること

- **原典の崩壊の防ぎ方**: I-JEPA・V-JEPA・V-JEPA 2 の target は、EMA のエンコーダに sg をかけたもの。V-JEPA §3.1 は理由として「EMA によって予測器がエンコーダより速く動き、最適に近いまま保たれる」を挙げる（BYOL の議論）。公式コード（ijepa・jepa）は target のトークンに affine なしの `F.layer_norm` をかけてから損失を取る。これは論文に書かれていない（コードからの記憶で、未確認）。私たちは最後の stage に affine つきの LayerNorm を置き、それを target にしている。gain を 0 に縮めれば、トークンはクリップの中で定数になれる
- **LeJEPA の主張の範囲**: 「予測器と teacher-student は、非対称な情報がないときは主に崩壊を防ぐ役で、SIGReg があれば外せる」（§6.1）。主張しているのは view 不変の形だけで、トークンのマスク予測では試していない。検索でも、I-JEPA・V-JEPA の EMA を SIGReg で置き換えた報告は見つからなかった。近い follow-up は、多モーダルの fusion トークンに SIGReg をかける Le MuMo JEPA（2603.24327）
- **EMA の代わりの公表例**: SALT（2509.24317）。画素の再構成で学んだ教師を凍らせ、生徒がマスクした領域の教師の潜在を当てる。弱い教師からでも良い生徒ができ、生徒の損失と probe の相関は $R^2 > 0.95$。ただし教師は画素の再構成で学ぶので、M17 の規則（再構成の項なし）とは合わない。学習前の ViT を凍った教師にする版なら規則の中に収まる。SALT は弱い教師でも効くと報告しているので、診断の候補になる
- **C-JEPA（2410.19560）**: I-JEPA の EMA は崩壊を完全には防がず、パッチ表現の平均を正しく学べない、という指摘。VICReg の分散・共分散の項を足して直している。私たちの「予測器が平均を出す」と同じ型の問題
- **私たちの失敗の型と、その仕組み**:
  1. トークンの近道: EMA も sg もないので、予測の損失の勾配が target 側にも流れる。target のトークンを「位置の符号（入力に足した 3D sincos が残差でそのまま出る）＋クリップの平均（色）」にすると、予測器は位置と context の平均から完全に当てられる。SIGReg は pool した $z$（＝クリップの平均）にしかかからない。`both` でもトークンの項は $S_t/T$ で 512 分の 1 に薄まる。sg を入れるとランクは 16 に上がったが（診断）、target は同じ重みのままで動き続ける（SimSiam 型）ので、これだけでは足りない
  2. 色・場面の近道: view はどれも同じクリップの同じ色を見ている。B=32 のストリームは 9 本の動画に散らばるので、SIGReg は「どの動画・どの場面か」で満たせる。λ を上げると、学ぶ側の `recon_fit` は 0.71→0.37 に下がるのに、val は悪くなった。これは場面の見分けを覚えていることを示す。LeJEPA は view ごとの色の拡張（grayscale も）で、この近道を消している
  3. 最良のチェックポイントが warmup の終わり（300 秒）なのは、lr の高さより、近道へ流れていく目的の式と合う。ただし lr 1e-5〜3e-5・wd・減衰の効果はまだ切り分けていない

## 5. 私たちとの違い（不具合の候補、疑わしい順）

| # | 違い | 原典 | 確かめ方 |
| -- | -- | -- | -- |
| 1 | トークンの target が EMA なし・sg なし・LayerNorm なし。SIGReg は $z$ だけ | EMA + sg + LN（I/V-JEPA）。LeJEPA はトークンを当てない | (a) 予測器の復元の誤差を、同じクリップの見えるトークンの平均（`completion.py` の `_vis`）と比べる。(b) クリップの中のトークンの分散 ÷ 全体の分散、位置の平均を引いた後の分散。(c) target に affine なしの LN、sg を入れて比べる |
| 2 | 拡張なし、9 本の動画、B=32 の i.i.d. でないストリーム | multi-crop・色の拡張（LeJEPA）、resize crop（V-JEPA）、batch 2048〜3072 の i.i.d. | (a) view ごとの色の拡張を入れる。(b) `task=offline` で、9 本から i.i.d. に引いた B=64〜128 のバッチで BP を回し、ストリームの相関を切り分ける |
| 3 | weight decay 0 | 0.04〜0.4（I/V-JEPA）、5e-2（LeJEPA） | wd 0.05 |
| 4 | warmup の後は lr が一定、減衰なし。Warmup は moment の時間も伸ばす | cosine（I/V-JEPA、LeJEPA）、warmup-constant + cooldown（V-JEPA 2） | 600 秒の診断を cosine で（`eta_min` = 1e-3 倍）。warmup は 10%（60 秒） |
| 5 | lr 7.5e-5/step @ B=32 | 平方根則で 6.4e-5 | {1e-5, 3e-5, 1e-4} |
| 6 | 予測器 2 層・幅 256 | 6〜12 層・幅 384（I-JEPA Table 12・14） | 深さ 4〜6、幅 128〜192 のボトルネック |
| 7 | val の SIGReg のバッチが、1 本の動画の数分（Zurich を約 1.4 時間とみて 3〜4 分）から来る | — | val の窓を混ぜてからバッチにする。または $N$ と出どころをそろえた学ぶ側の窓と比べる |
| 8 | SIGReg 256 方向、projector なしの 256 次元 | 1024 方向推奨、projector（3 層、出力 16〜128 次元）の上 | projector 64 次元の上に SIGReg |

## 6. 提案: 時空間の補完の目的（再構成なし）

### 6.1 式

クリップ $x$（$N=16$、$\Delta$=3 フレーム、$\tau=2$、時間 8 × 空間 64 = 512 トークン）ごとに、同じエンコーダ $f$ で次の view を別々に符号化する（packed の区間の中だけで attention）。

| view | 見えるトークン | 数（$N=16$） | 補う相手 |
| -- | -- | -- | -- |
| $g$（global） | 全部 | 512 | — |
| $c_s$（短い範囲） | V-JEPA の 8 × 0.15 の tube の外 | 18 × 8 = 144 | 隠した空間（全時間） |
| $c_l$（長い範囲） | 2 × 0.7 の tube の外 | 6 × 8 = 48 | 大きく隠した空間 |
| $c_p$（過去） | 時間の slot 0〜5（最初の 12 フレーム）で、さらに 8 × 0.15 の空間のマスク | 18 × 6 = 108 | 未来の 2 slot（0.3 秒）と、隠した空間 |
| （任意）$c_r$（空間の切り出し） | 4×4〜6×6 patch の 1 矩形、全時間 | 128〜288 | 切り出しの外 |

view ごとに $e_v = \mathrm{mean}_{t \in v} f(x)_t$（256 次元、backbone の pool）と $u_v = h(e_v)$（projector。Linear-LN-GELU を 3 層、512→512→64）を作り、

$$
L = (1-\lambda)\Big[\underbrace{\tfrac{1}{V}\textstyle\sum_v \lVert u_g - u_v\rVert^2}_{z\ \text{の水準の補完（LeJEPA Eq. 8、}V_g=1)} + \beta\, L^{\rm tok}\Big] + \frac{\lambda}{V}\sum_v \mathrm{SIGReg}(\{u_{n,v}\}_n)
$$

とする。

- **$z$ の水準の補完**: 部分の view の $u_v$ は、見えない空間と未来を含む全体の $u_g$ に一致しなければならない。だから「画像の一部が、その外を補完する特徴」を、$z$ が直接持つように押される。次フレームだけを当てる課題と違い、空間の補完と未来の補完を同じ式で扱う（V-JEPA Table 4 の multi-block > causal の結果と合う）
- **予測器なしから始める**: LeJEPA の Table 4 では予測器は効かない。ただ私たちの view は情報の量が非対称（過去だけの view は未来を見ない）なので、軸として「view の種類で条件づけた小さな MLP の予測器 $p(u_v, \mathrm{kind}_v) \to u_g$」を置く。予測器があると、$u_g$ は部分から予測できる成分だけに縮まずにすむ
- **トークンの項 $L^{\rm tok}$（補助、$\beta \in \{0, 0.5, 1\}$）**: V-JEPA のマスク予測を、$c_s$・$c_l$・$c_p$ の隠したトークンに。target は global view のトークンに affine なしの LayerNorm をかけたもの。位置の平均を batch で引いた後のトークンにも SIGReg（$1/T$ にせず、トークンを sample として数える）をかけ、クリップの中の崩壊を止める。$\beta=0$ が LeJEPA そのもので、最初の診断はこれから
- **色の近道を塞ぐ**: view ごとに独立な色の拡張（ColorJitter 0.4/0.4/0.2/0.1 p=0.8、grayscale p=0.2。1 つの view の中では全フレームに同じ変換をかけ、時間の情報を保つ）。強さは軸 {なし、弱、LeJEPA}。色を強く消すと、posthoc の再構成は色の分だけ悪くなる。ユーザーの方針（再構成を重視しすぎない）とは両立する。拡張はホスト（`StackChunker`）で行うので、局所性に触れない
- **SIGReg**: view ごとに、projector の出力 64 次元へ。方向は 256〜1024、step ごとに引き直す（今のまま）。λ は 0.05（論文）と 0.02（minimal）。B=32 は推奨の 128 より小さいが、偏りは $O(1/N)$ で batch 16 でも小さい（§5.1）
- **posthoc の特徴量 $z$**: global view の backbone の pool $e_g$（256 次元、$D \le 1024$）。LeJEPA の probe と同じく、projector の前で取る。今の規則（$z$ は pool した 1 本のベクトル）を守る

### 6.2 始めのハイパラ（BP、600 秒の診断）

- AdamW、peak の lr 1e-4/frame（約 7.5e-5/step）。軸は {3e-5, 1e-4, 3e-4}
- warmup 60 秒（10%）、その後 cosine で peak の 1e-3 倍まで。継続学習の本番では V-JEPA 2 の warmup-constant にする
- weight decay 0.05（固定）
- $\tau_m$ 0.24 s・$\tau_v$ 25 s（step 25 ms で β = (0.9, 0.999)）。今の 0.3・30 とほぼ同じなので、変えなくてもよい
- λ 0.05、projector 3 層・出力 64、SIGReg 512 方向
- マスクは V-JEPA のまま。過去の view は最初の 6 slot
- B=32（メモリの上限）。i.i.d. の対照として `task=offline` で B=64

### 6.3 packed の形への写し方

- 今の packed は `[target T | ctx_1 K_1 | ctx_2 K_2]` を 1 つの sample に並べ、`segments` で attention を区間に閉じている。これに過去の view の区間を足すと `segments = (512, 144, 48, 108)`、$L$ = 812（今は 704。メモリは約 1.15 倍）
- 変える場所: (1) `masks.pack_views` は tube（全 slot に同じ空間のマスク）を前提にしている。時間 × 空間のマスク `(n, T_t, gh, gw)` を受け取れるようにする。過去の view はこれで作る（`completion.py` の `future_tokens` と同じ考え方）。(2) `PackedJEPAHead`・`JEPABP` の損失で、区間ごとに平均の pool → projector → $u_v$、$L_z$ と view ごとの SIGReg を計算する。projector はヘッドのパラメタ。(3) 色の拡張は `StackChunker` で view ごとに（target と各 context に別の変換）。ただしパッチは target と context で共有しているので、view ごとに画素を作り直す必要がある。`pack_views` の前で view ごとに積みを複製する
- 局所性: 損失はすべてヘッドのノードの中にあり、credit は今と同じ $-J^\top\delta$ のメッセージだけで stage をまたぐ。ノードをまたぐ `.backward()` やバリアは要らない

### 6.4 指標

- **主な指標（held-out、Zurich）**:
  - $z$ の水準の補完の誤差 $E_z^{(k)} = \overline{\lVert u_g - u_{c_k}\rVert^2} / \overline{\lVert u_g - \bar u_g\rVert^2}$（view の種類 $k$ = 短・長・過去ごと。尺度によらない。1 は「全体の平均を当てた」に等しい）
  - トークンの補完の誤差（`measure/completion.py` の $E = 1 - R^2$、`short`・`long`・`future`）。`_vis`（同じクリップの見えるトークンの平均）を下回ることを求める。クリップの中で正規化した版（分母を「そのクリップの target の平均」との差にする）も足すとよい。これが 1 に近いと、予測器は平均を出しているだけ
- **一緒に見るもの**: SIGReg（val の窓を混ぜてから学ぶ側と同じ $N$ で）と実効ランク。LeJEPA Fig. 10 のように「補完の誤差 × 等方性」の平面で読む。補完の誤差だけなら崩壊でも小さくできる
- posthoc の再構成・線形の動力学は今のまま残す。ただし判定の重みは下げる（ユーザーの方針）

## 7. 進め方の案

1. 基準の課題（§3、S1〜S3）。std の 4090 1 台で、1 段 1 時間以内。不具合が私たちの実装（ViT・SIGReg・最適化）にあるのか、目的の式にあるのかを切り分ける
2. Walking Tours の BP の 600 秒の診断（2 seed）: 今の preset ＋ wd 0.05 ＋ cosine（ハイパラだけを直す） / ＋ target の LN と sg / ST-LeJEPA で $\beta=0$、拡張なし / 同じく拡張あり / `task=offline` で i.i.d. の B=64。指標は §6.4
3. 上で学習前の ViT を補完の誤差と posthoc で下回る設定が出てから、study と PC-ALM に進む

## 出典

- I-JEPA: arXiv 2301.08243、§3（Fig. 4・マスク）、App. A（最適化・予測器）、App. C Table 12〜14
- V-JEPA: arXiv 2404.08471、§3.1（Eq. 1・2、EMA の理由）、§4 Table 4（マスクの課題）、App. B、App. C Table 8、App. E.4（Fig. 8）
- V-JEPA 2: arXiv 2506.09985、§2.1・§2.4、App. A Table 9・10・12
- LeJEPA: arXiv 2511.08544v3、§4.3、§5.1（Eq. 6〜9、Theorem 6）、§6.1（Table 1・Fig. 8・9、Experiment Details 1）、§6.2（Eq. 10）、§6.3、Table 3・4・5・7
- LeJEPA の minimal: https://github.com/rbalestr-lab/lejepa の `MINIMAL.md`（2025-11-20）と README
- SALT: arXiv 2509.24317。C-JEPA: arXiv 2410.19560（NeurIPS 2024）。Le MuMo JEPA: arXiv 2603.24327（概要だけ読んだ）
- 私たちの設定: `exp/video/conf/experiment/m17/wt_n16.yaml`、`conf/optim/adamw.yaml`、`schema/learn.py`・`schema/run.py`（`learn_interval` 25 ms）、`dyn/nets.py::make_optimizer`、`exp/lib/optim.py::Warmup`、`jepa/loss.py::jepa_loss_packed`、`programs/vision.py`（val の窓とバッチ）、`data/walking_tours.py::eval_clips`
