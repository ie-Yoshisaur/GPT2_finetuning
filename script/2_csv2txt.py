import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small")

csv_path = "./csv/processed_csv/input_output.csv"
train_df = pd.read_csv(csv_path)

with open("./txt/gpt2_train_data.txt", 'w') as output_file:
    tmp = []
    for x, y in zip(train_df["input"], train_df["output"]):
        x = x.lower()
        y = y.lower()
        inp_tokens = tokenizer.tokenize(x)[:256]
        inp = "".join(inp_tokens).replace('▁', '')
        out_tokens = tokenizer.tokenize(y)[:256]
        out = "".join(out_tokens).replace('▁', '')

        data = "<s>" + inp + "[SEP]" + out + "</s>"
        tmp.append(data)
    txt = "".join(tmp)*10
    output_file.write(txt + '\n')
