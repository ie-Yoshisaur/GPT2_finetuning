rm -rf transformers output/*
git clone -b v4.4.2-release https://github.com/huggingface/transformers/
rm ./transformers/examples/language-modeling/run_clm.py
cp run_clm.py ./transformers/examples/language-modeling/run_clm.py
pip3 install --no-cache-di -r requirements.txt
