import torch
from transformers import BertJapaneseTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
from datasets import load_dataset

# ====== ファイルパスを指定 ======

# 学習・検証データを指定
data = "SNOW"     # 学習・検証データにSNOWを使用する場合
#data = "SNOW+MATCHA"   # 学習・検証データにMATCHAを使用する場合

train_file = f"../../../data/{data}/train.csv"  # 学習データファイル (CSV形式)
val_file = f"../../../data/{data}/valid.csv"    # 検証データファイル (CSV形式)

# ====== データセットの読み込み ======
dataset = load_dataset(
    "csv",
    data_files={"train": train_file, "validation": val_file},
    column_names=["text", "label"]  # カラム名を明示的に指定
)
# ====== 学習用設定 ======

# ====== トークナイザーとモデルの準備 ======
#追加事前学習無しのBARTモデルを使用
pretrained_model = "tohoku-nlp/bert-base-japanese-whole-word-masking"  # 日本語BARTモデル
tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)

model = EncoderDecoderModel.from_encoder_decoder_pretrained(pretrained_model, pretrained_model)

# デコーダの開始トークンを設定
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# ====== トークナイズ関数 ======
def preprocess_function(examples):
    # 入力テキストとターゲットテキストをトークン化
    inputs = tokenizer(examples["text"], max_length=128, truncation=True, padding="max_length")
    targets = tokenizer(examples["label"], max_length=128, truncation=True, padding="max_length")
    
    # ラベルとして targets の input_ids を設定
    inputs["labels"] = targets["input_ids"]
    return inputs

# データセットをトークナイズ

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])

# ====== トレーニング設定 ======
training_args = TrainingArguments(
    output_dir=f"./../../../model/train_{data}_BART",# 出力ディレクトリ
    eval_strategy="epoch",                        # 評価のタイミング
    learning_rate=5e-5,                           # 学習率
    per_device_train_batch_size=16,               # トレーニングのバッチサイズ
    per_device_eval_batch_size=16,                # 評価のバッチサイズ
    num_train_epochs=10,                          # エポック数
    weight_decay=0.01,                            # 重み減衰
    save_strategy="epoch",                        # モデル保存のタイミング
    logging_dir="./logs",                         # ログの出力先
)

# ====== Trainerの設定 ======
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# ====== モデルの学習 ======
trainer.train()

# ====== 推論用関数 ======
def simplify_text(text):
    # モデルをGPU（CUDA）に移動
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 入力テキストをトークン化してデバイスに移動
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # デバイスに移動
    
    # モデルによる生成（decoder_start_token_idを直接指定）
    outputs = model.generate(
        **inputs, 
        max_length=128, 
        num_beams=4, 
        early_stopping=True,
        decoder_start_token_id=model.config.decoder_start_token_id  # 明示的に設定
    )
    # トークンをデコードしてテキストに変換
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ====== テスト ======

test_text = "これは非常に高度な技術が求められる作業です。"
simplified_text = simplify_text(test_text)
print("Original Text:", test_text)
print("Simplified Text:", simplified_text)
