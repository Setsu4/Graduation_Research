import MeCab
import os
import pandas as pd

def analyze_text(text):
    """
    テキストを形態素解析して、表層形のみをスペース区切りで返す関数

    Args:
        text (str): 解析するテキスト

    Returns:
        str: 形態素解析の結果（表層形をスペース区切りで返す）
    """
    tagger = MeCab.Tagger("-r /dev/null -d /home/ito/local/lib/mecab/dic/ipadic")
    node = tagger.parseToNode(text)
    results = []

    while node:
        if node.surface:  # 空白のノードを除外
            results.append(node.surface)
        node = node.next

    return ' '.join(results)

def check_file_exists(filepath):
    """
    ファイルが存在するか確認し、存在しない場合はエラーをスローする関数

    Args:
        filepath (str): 確認するファイルパス
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"ファイルが存在しません: {filepath}")
    print(f"ファイル確認済み: {filepath}")

def check_dir_exists(dirpath):
    """
    ディレクトリが存在するか確認し、存在しない場合は作成する関数

    Args:
        dirpath (str): 確認するディレクトリパス
    """
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    print(f"ディレクトリ確認済み: {dirpath}")

def process_file(input_file, output_file):
    """
    入力CSVファイルを読み込み、形態素解析を行い、結果を出力CSVファイルに保存する関数

    Args:
        input_file (str): 入力CSVファイルのパス
        output_file (str): 出力CSVファイルのパス
    """
    # 入力ファイル確認
    check_file_exists(input_file)

    # 出力ディレクトリ確認
    check_dir_exists(os.path.dirname(output_file))

    # 入力ファイルを読み込み
    data = pd.read_csv(input_file)

    # 各列に形態素解析を適用
    for column in ['text', 'label', 'simple']:
        if column in data.columns:
            data[column] = data[column].apply(analyze_text)

    # 出力ファイルに保存
    data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"処理結果を保存しました: {output_file}")

def process_files():
    input_dir = "./../train_result_data/results_MSimple_BART"
    output_dir = "./../mecab_data/mecab_MSimple_BART"

    file_pairs = [
        ("results_SNOW_test.csv", "mecab_SNOW_test.csv"),
        ("results_MATCHA_test.csv", "mecab_MATCHA_test.csv"),
        ("results_JADES_test.csv", "mecab_JADES_test.csv"),
    ]

    for input_file, output_file in file_pairs:
        input_path = f"{input_dir}/{input_file}"
        output_path = f"{output_dir}/{output_file}"

        print(f"Processing {input_path} -> {output_path}...")
        process_file(input_path, output_path)

    print("全てのファイルの形態素解析が完了しました。")

if __name__ == "__main__":
    process_files()
