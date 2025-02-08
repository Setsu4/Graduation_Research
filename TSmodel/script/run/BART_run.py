import os
import torch
import pandas as pd
import re
from transformers import BertJapaneseTokenizer, EncoderDecoderModel

# ====== トークナイザーとモデルの準備 ======
trained_model = "../../model/train_SNOW_BART/checkpoint-70"  # 学習済みモデルのパス
if not os.path.exists(trained_model):
    raise FileNotFoundError(f"モデルディレクトリが存在しません: {trained_model}")
print(f"モデルディレクトリ確認済み: {trained_model}")

tokenizer = BertJapaneseTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-whole-word-masking")
model = EncoderDecoderModel.from_pretrained(trained_model)

# デコーダの開始トークンとパディングトークンを設定
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# ====== 推論用関数 ======
def simplify_text(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # すべてのスペースを削除
    edited_text = re.sub(r'\s+', '', simplified_text)
    
    return edited_text

# ====== ファイル存在確認関数 ======
def check_file_exists(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"ファイルが存在しません: {filepath}")
    print(f"ファイル確認済み: {filepath}")

def check_dir_exists(dirpath):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    print(f"ディレクトリ確認済み: {dirpath}")

# ====== CSVファイルの処理 ======
def process_file(input_file, output_file):
    # 入力ファイル確認
    check_file_exists(input_file)

    # 出力ディレクトリ確認
    output_dir = os.path.dirname(output_file)
    check_dir_exists(output_dir)

    # CSVファイルを読み込み
    df = pd.read_csv(input_file)

    # 平易化処理と標準出力
    results = []
    for i, row in df.iterrows():
        original_text = row["text"]
        simplified_text = simplify_text(original_text)
        results.append(simplified_text)

        # 平易化する文とその結果を標準出力
        print(f"Original: {original_text}")
        print(f"Simplified: {simplified_text}")
        print("-" * 50)

    # 平易化結果をデータフレームに追加
    df["simple"] = results

    # 結果をCSVに保存
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# ====== 実験の実行 ======
output_dir = "./../../../output_result/run_result/run_BART"
process_file("../../../evaluation/data/SNOW_test.csv", f"{output_dir}/run_SNOW_test.csv")
process_file("../../../evaluation/data/MATCHA_test.csv", f"{output_dir}/run_MATCHA_test.csv")
process_file("../../../evaluation/data/JADES_test.csv", f"{output_dir}/run_JADES_test.csv")
