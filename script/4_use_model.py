from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, T5Tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-small')
tokenizer.do_lower_case = True

output = Path('output/')
model = AutoModelForCausalLM.from_pretrained(output)
model.to(device)
model.eval()

def generate_reply(inp, num_gen=1):
    input_text = "<s>" + str(inp) + "[SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    out = model.generate(input_ids, do_sample=True, max_length=128, num_return_sequences=num_gen, top_p=0.95, top_k=20, bad_words_ids=[[1], [5]], no_repeat_ngram_size=3)
    for sent in tokenizer.batch_decode(out):
        print(sent)

while(True):
    msg = input()
    generate_reply(msg)
