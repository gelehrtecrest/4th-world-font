# FF14 第四世界文字 フォント生成ツール (FF14 Fourth World Font Generator)

FF14（Final Fantasy XIV）のゲーム内に登場する「第四世界文字」の画像データから、背景除去・ノイズクリーニング・ベクター（SVG）変換、およびインストール可能な TrueType フォント（`.ttf`）を自動生成するツールセットです。

FF14はスクウェア・エニックス社の登録商標です。
© SQUARE ENIX

---

## 📁 ディレクトリ構成

```text
4th-world-font/
├── data/                       # 入力画像フォルダ（文字画像を追加する場所）
│   ├── large_i.jpg             # 大文字 I (large_*.jpg または large-*.jpg)
│   ├── large_u.jpg             # 大文字 U
│   ├── small_d.jpg             # 小文字 d (small_*.jpg または small-*.jpg)
│   └── ...
├── output/                     # 自動生成成果物
│   ├── fonts/
│   │   └── FF14FourthWorld.ttf # 生成されたTrueTypeフォント（Windowsにインストール可能）
│   ├── svg/                    # 各文字の高解像度ベクターSVGファイル
│   ├── cleaned_png/            # 背景除去・ノイズ除去済みの二値化PNG
│   ├── preview.html            # ブラウザで確認できる対話型プレビューダッシュボード
│   ├── sample_render.png       # フォントレンダリング見本画像
│   └── compare_levels.png      # スムージング強度比較チャート (Level 0〜100)
├── vectorize_pipeline.py       # メイン生成パイプライン
└── README.md                   # 本ドキュメント
```

---

## 🚀 使い方

### 1. 新しい文字画像の追加
`data` フォルダに判明した文字画像を追加します。
ファイル名は以下の規則に対応しています：
- 大文字: `large_*.jpg` または `large-*.jpg` （例: `large_a.jpg`, `large-b.png`）
- 小文字: `small_*.jpg` または `small-*.jpg` （例: `small_a.jpg`, `small-b.png`）

### 2. パイプラインの実行（スムージング強度の指定）
PowerShell またはコマンドプロンプトで以下を実行します：

```bash
# デフォルト（おすすめの中間設定: Level 50）
python vectorize_pipeline.py

# お好みの強度（0〜100）を指定して実行
python vectorize_pipeline.py --smooth 50
python vectorize_pipeline.py -s 30
python vectorize_pipeline.py -s 70
```

#### スムージング強度（`--smooth` / `-s`）の目安
| レベル | 特徴 |
| :--- | :--- |
| **`0` (Raw)** | **元画像に100%忠実**。ピクセルの微小な凹凸やテクスチャの揺らぎもそのまま残します。 |
| **`50` (Default)** | **推奨のバランス設定**。原画特有の筆跡・セリフ・抑揚をしっかり残しつつ、不快なささくれやジャギーのみを除去します。 |
| **`100` (Smooth)** | **最大平滑化**。極めて滑らかで幾何学的なラインにします。 |

> [!TIP]
> 各レベルの見た目の変化は、[`output/compare_levels.png`](file:///D:/workspace/github/4th-world-font/output/compare_levels.png) または [`output/preview.html`](file:///D:/workspace/github/4th-world-font/output/preview.html) のチャートで視覚的に比較できます。

### 3. フォントの利用
- **PCへのインストール**:
  [`output/fonts/FF14FourthWorld.ttf`](file:///D:/workspace/github/4th-world-font/output/fonts/FF14FourthWorld.ttf) を右クリックして「インストール」を選択すると、Word、Photoshop、Illustrator、ブラウザなどで本物のフォントとして入力できます。
- **Web上でのプレビュー**:
  [`output/preview.html`](file:///D:/workspace/github/4th-world-font/output/preview.html) をブラウザで開くと、元画像・二値化画像・SVGの比較確認や、自由に入力して試し打ちができるテキストボックスが利用できます。
