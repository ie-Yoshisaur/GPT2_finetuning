from pathlib import Path

output = Path("output/")

import torch
from  transformers import AutoModelForCausalLM, T5Tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-medium')
tokenizer.do_lower_case = True
model = AutoModelForCausalLM.from_pretrained(output)
model.to(device)
model.eval()

def generate_reply(inp, num_gen=1):
    input_text = "<s>" + str(inp) + "[SEP]"
    input = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input, do_sample=True, max_length=128, num_return_sequences=num_gen, top_p=0.95, top_k=50, bad_words_ids=[[1], [5]], no_repeat_ngram_size=3)
    print(tokenizer.batch_decode(output))

while(True):
    msg = input()
    generate_reply(msg)
