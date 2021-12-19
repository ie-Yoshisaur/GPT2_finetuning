pip3 install -r requirements.txt
python3 script/1_process_csv_files.py
python3 script/2_csv2txt.py
rm -rf transformers
git clone https://github.com/huggingface/transformers.git
bash script/3_gpt2_finetuning.sh
python3 script/4_use_model.py
