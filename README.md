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

