# Graduation_Research

##  概要

テキスト平易化モデルの学習とSARIを用いた評価

卒業研究成果物

記載者：千葉工業大学先進工学部未来ロボティクス学科藤江研究室 伊藤雪

### 使用している既存モデルとパッケージ
平易化モデルの学習には既存の言語モデルを用いる．

平易化結果の評価にはPython3のパッケージである[EASSE](https://github.com/feralvam/easse)を用いる．

#### BART
東北大学自然言語処理研究グループが開発したBARTを使用
日本語Wikipediaで事前学習済み

[huggingface](https://huggingface.co/tohoku-nlp/bert-base-japanese-whole-word-masking)にて公開されているモデルを使用．

#### EASSE
テキスト平易化の自動評価を容易にすることを目的としたPython3パッケージ

https://github.com/feralvam/easse

## レポジトリ

### LICENSE
Apache License2.0を使用している。使用しているDETRとBLIPのLICENSEは下記の通りになっている

DETR：Apache License 2.0

BLIP：BSD 3-Clause License

### .gitignore
画像分類実行時にモデルの特徴量ファイル(ptファイル)と学習済みモデルの重みの保存にpthファイルを作成している。ファイル容量が大きいため、gitignoreとした。

### object_detection
DETRの物体検出精度を評価するために既存モデルで物体検出を実行

### detr
DETRの中間特徴量を使用した画像分類

### blip
BLIPの中間特徴量を使用した画像分類

### blip+detr
DETRとBLIPを統合した特徴量で画像分類

## 実行方法
各コードの実行方法はフォルダ先のREADMEに記載してあります。


##  インストール
```sh
git clone git@github.com:shun-ski/graduation_thesis.git
```

## 画像分類結果
| モデル名       | 正答率   |
|-------------|---------------------------|
| BLIP   | 61%   |
| DETR| 43%|
|BLIP＋DETR|77%|


## 開発環境
Python：3.10.12

PyTorch：2.0.1

WSL：Ubuntu-22.04

## 謝辞
卒業研究でご指導いただいた藤江真也教授、卒論審査でお世話になった未来ロボティクス学科の先生方、研究のアドバイスをくれた研究室の仲間たちに感謝申し上げます。