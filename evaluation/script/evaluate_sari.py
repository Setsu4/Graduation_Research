from easse.sari import corpus_sari
import pandas as pd
import os
import MeCab

# ====== 評価を実行する関数 ======
def tokenize_text(text):
    mecab = MeCab.Tagger("-r /dev/null -d /home/ito/local/lib/mecab/dic/ipadic")
    return mecab.parse(text).strip()

def process_file(input_file, output_file):
    """
    入力ファイルを読み込み、SARIスコアを計算し、結果を出力ファイルに保存する関数

    Args:
        input_file (str): 入力CSVファイルのパス
        output_file (str): 出力CSVファイルのパス
    """
    # 入力ファイルの存在確認
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"ファイルが存在しません: {input_file}")
    print(f"入力ファイル確認済み: {input_file}")

    # データの読み込み
    data = pd.read_csv(input_file)

    # SARIスコアを計算して全行に追加
    sari_scores = []
    for _, row in data.iterrows():
        orig_sent = tokenize_text(row['text'])
        sys_sent = tokenize_text(row['simple'])
        ref_sent = tokenize_text(row['label'])

        # 参照文をリスト形式に変換
        refs_sents = [[ref_sent]]

        # SARIスコアの計算
        sari_score = corpus_sari(orig_sents=[orig_sent], sys_sents=[sys_sent], refs_sents=refs_sents)
        sari_scores.append(sari_score)

    # SARIスコアをデータフレームに追加
    data['SARI'] = sari_scores

    # 全体の平均スコアを計算
    mean_sari_score = sum(sari_scores) / len(sari_scores)

    # 平均スコアを新しい行として追加
    mean_row = pd.DataFrame({
        'text': ['平均 (全体)'],
        'label': ['-'],
        'simple': ['-'],
        'SARI': [mean_sari_score]
    })
    data = pd.concat([data, mean_row], ignore_index=True)

    # 出力ディレクトリの確認と作成
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print(f"出力ディレクトリ確認済み: {output_dir}")

    # 結果をCSVに出力
    data.to_csv(output_file, index=False)
    print(f"評価結果を {output_file} に保存しました。")

# ====== 実験の実行 ======
# 評価をしたいモデルを選択
#model = "BART"
#model = "BART-MATCHA"
#model = "Simple"
model = "Simple-MATCHA"

input_dir = f"./../../output_result/run_result/run_{model}"
output_dir = f"./../../output_result/evaluate_result/SARI_{model}"
process_file(f"{input_dir}/run_SNOW_test.csv", f"{output_dir}/SARI_SNOW_test.csv")
process_file(f"{input_dir}/run_MATCHA_test.csv", f"{output_dir}/SARI_MATCHA_test.csv")
process_file(f"{input_dir}/run_JADES_test.csv", f"{output_dir}/SARI_JADES_test.csv")
