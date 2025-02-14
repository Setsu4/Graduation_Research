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

## リポジトリ

### LICENSE
使用しているBARTとEASSEのLICENSEは下記の通りになっている

BART：Creative Commons Attribution-ShareAlike 3.0

EASSE：GNU General Public License v3.0

### SimpleBART
既存のBARTに対して，平易化に特化した事前学習を行う．

### TSmodel
#### train
既存のBARTおよび，追加事前学習を行ったBARTに対して，平易化モデル構築のための学習を行う．

#### run
trainで学習したモデルを使用し，評価用データに対して平易化を行う．

### evaluation
TSmodelで出力した平易化結果に対して，SARIをもいいた評価を行う．

### output_result
各種実行結果が保存されます．

#### run_result
評価データに対して平易化を行った結果が保存されます．

#### evaluate_result
run_resultの内容に加え，評価値を出力したデータが保存されます．

## 実行方法
各コードの実行方法は[こちら](https://docs.google.com/document/d/1CylmH0FqLUmR5OFeo7YzBWaL_-Me6VHIwvcG76cHJCI/edit?usp=sharing)に記載しています．


##  インストール
```sh
git clone https://github.com/Setsu4/Graduation_Research.git
```

## 開発環境
### モデルの学習
Python：3.9.20

torch：2.5.1

silver10

### 平易化結果の評価
python：3.6.13

silver10

## 謝辞
卒業研究をご指導くださった藤江真也教授，卒論審査でお世話になった未来ロボティクス学科の先生方，そして貴重なアドバイスをくださった研究室の仲間たちに，心より感謝申し上げます．