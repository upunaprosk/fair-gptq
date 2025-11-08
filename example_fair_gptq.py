# This is an example sctipt of using fair-gptq

import random
import os
import shutil
import numpy as np
import torch
from typing import Any
import json
import urllib.request
from gptqmodel import GPTQModel, QuantizeConfig

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Set random seed
seed_everything(0)

# Load StereoSet data
url = "https://raw.githubusercontent.com/gsgoncalves/EMNLP2023_llm_compression_and_social_bias/refs/heads/main/data/stereoset/dev.json"
with urllib.request.urlopen(url) as response:
    data = json.load(response)

examples = []
for entry in data['data']['intrasentence']:
    for sentence_entry in entry['sentences']:
        sentence = sentence_entry['sentence']
        gold_label = sentence_entry['gold_label']
        if gold_label in ['stereotype', 'anti-stereotype']:
            examples.append(sentence)

# Prepare logging directory
if os.path.exists("logs"):
    shutil.rmtree("logs")
os.makedirs("logs", exist_ok=True)

print(examples[:2])

# Quantization configuration
alpha = 0.5
models = ["facebook/opt-6.7b"]
string_all_ignore = "k_proj,q_proj,v_proj,fc1,out_proj,fc2"

for l in ["0,1", "31,30"]:
    for m in models:
        l_print = l.replace(",", "-")
        quant_path = f"{m.split('/')[-1]}-int4-fair-gptq-{l_print}"

        if os.path.exists(quant_path):
            shutil.rmtree(quant_path)

        quant_config = QuantizeConfig(
            bits=4,
            group_size=128,
            alpha=alpha,
            desc_act=False,
            select_layer=str(l)
        )

        model = GPTQModel.load(m, quant_config)
        model.quantize(examples, batch_size=2)
        model.save(quant_path)

        os.makedirs("logs", exist_ok=True)
        model = GPTQModel.load(quant_path)

        result = model.generate("Uncovering deep insights begins with")[0]
        print(model.tokenizer.decode(result))

        repo_id = f"USERNAME/{quant_path}"
        commit_message = f"Quantized model for {m}"
        # Copy logs
        for log_file in os.listdir("logs"):
            src = os.path.join("logs", log_file)
            dst = os.path.join(quant_path, log_file)
            shutil.copy(src, dst)

        GPTQModel.push_to_hub(
            repo_id=repo_id,
            quantized_path=quant_path,
            exists_ok=True,
            private=False
        )

        # Clear logs
        for log_file in os.listdir("logs"):
            os.remove(os.path.join("logs", log_file))