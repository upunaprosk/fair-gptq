import os
import random
import shutil
import json
import urllib.request
import argparse
import numpy as np
import torch
from gptqmodel import GPTQModel, QuantizeConfig

def seed_everything(seed: int = 0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_stereoset_data() -> list[str]:
    url = (
        "https://raw.githubusercontent.com/"
        "gsgoncalves/EMNLP2023_llm_compression_and_social_bias/"
        "refs/heads/main/data/stereoset/dev.json"
    )
    with urllib.request.urlopen(url) as response:
        data = json.load(response)
    examples = []
    for entry in data["data"]["intrasentence"]:
        for sent_entry in entry["sentences"]:
            sentence = sent_entry["sentence"]
            gold_label = sent_entry["gold_label"]
            if gold_label in ["stereotype", "anti-stereotype"]:
                examples.append(sentence)
    return examples

def prepare_logs_dir(log_dir: str = "logs"):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

def quantize_model(
    model_name: str,
    select_layer: str,
    examples: list[str],
    bits: int,
    group_size: int,
    alpha: float,
    desc_act: bool,
    gradient_update: bool,
    ignore_bias: str,
    sum_hessians: bool,
    log_delta_w: bool,
    log_hessian_sensitivity: bool
):
    layer_print = select_layer.replace(",", "-")
    quant_dir = f"{model_name.split('/')[-1]}-int{bits}-fair-gptq-{layer_print}"
    if os.path.exists(quant_dir):
        shutil.rmtree(quant_dir)
    quant_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        alpha=alpha,
        desc_act=desc_act,
        select_layer=str(select_layer),
        ignore_bias=ignore_bias,
        gradient_update=gradient_update,
        sum_hessians=sum_hessians,
        log_delta_w=log_delta_w,
        log_hessian_sensitivity=log_hessian_sensitivity,
    )
    model = GPTQModel.load(model_name, quant_config)
    model.quantize(examples, batch_size=2, calibration_data_min_length=2)
    model.save(quant_dir)
    model = GPTQModel.load(quant_dir)
    result = model.generate("Uncovering deep insights begins with")[0]
    print(model.tokenizer.decode(result))
    if os.path.exists("logs"):
        for log_file in os.listdir("logs"):
            src = os.path.join("logs", log_file)
            dst = os.path.join(quant_dir, log_file)
            shutil.copy(src, dst)
    print(f"Saved quantized model: {quant_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--layers", type=str, default="11")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--desc_act", type=lambda x: x.lower() in ["true", "1", "yes"], default=False)
    parser.add_argument("--gradient_update", type=lambda x: x.lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--ignore_bias", type=str, default="k_proj,q_proj,v_proj,fc1")
    parser.add_argument("--sum_hessians", type=lambda x: x.lower() in ["true", "1", "yes"], default=False)
    parser.add_argument("--log_delta_w", type=lambda x: x.lower() in ["true", "1", "yes"], default=False)
    parser.add_argument("--log_hessian_sensitivity", type=lambda x: x.lower() in ["true", "1", "yes"], default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    examples = load_stereoset_data()
    prepare_logs_dir()

    for layer in args.layers.split(","):
        quantize_model(
            args.model,
            layer,
            examples,
            args.bits,
            args.group_size,
            args.alpha,
            args.desc_act,
            args.gradient_update,
            args.ignore_bias,
            args.sum_hessians,
            args.log_delta_w,
            args.log_hessian_sensitivity,
        )
    print("Done.")

if __name__ == "__main__":
    main()
