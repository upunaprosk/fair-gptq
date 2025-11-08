# Implementation tests of fair-gptq

This repository is structured as follows: 
- the code for fair-gptq is located in `gptq.py` and `config.py`.
- `./evaluation/` directory contains code for model evaluation.
- `./sanity-check/` directory contains code for sanity check experiments.
  
# Fair-GPTQ Code

We use the GPTQModel code and integrate a fairness term directly into the package's codebase.
See `example_fair_gptq.py` for a full usage example.

## Setting up for quantization
```bash
# Clone the GPTQModel repository and replace necessary files
git clone https://github.com/ModelCloud/GPTQModel.git
cd GPTQModel
git checkout v2.2.0
cd ..
cp gptq.py GPTQModel/gptqmodel/quantization/
cp config.py GPTQModel/gptqmodel/quantization/

# Install required packages
pip install -r requirements.txt
```

## Loading calibration data
To quantize the model, start by downloading the Stereoset Development subset available at:
https://raw.githubusercontent.com/gsgoncalves/EMNLP2023_llm_compression_and_social_bias/refs/heads/main/data/stereoset/dev.json

This dataset was released in the framework used by [Understanding the Effect of Model Compression on Social Bias in Large Language Models](https://aclanthology.org/2023.emnlp-main.161/) (Gonçalves & Strubell, EMNLP 2023).

```
import json
import urllib.request
url = "https://raw.githubusercontent.com/gsgoncalves/EMNLP2023_llm_compression_and_social_bias/refs/heads/main/data/stereoset/dev.json"
with urllib.request.urlopen(url) as response:
    data = json.load(response)
examples = []
for entry in data['data']['intrasentence']:  # Accessing 'intrasentence' which holds sentences
    sentences = entry['sentences']  # Access the 'sentences' key, which contains individual sentences
    # Iterate over the sentences to check their bias type and add to examples
    for sentence_entry in sentences:
        sentence = sentence_entry['sentence']  # Extract the sentence
        gold_label = sentence_entry['gold_label']  # Extract the bias label (gold_label)

        # Concatenate context with the sentence
        full_sentence =sentence

        # Add both stereotype and anti-stereotype sentences to examples
        if gold_label == 'anti-stereotype':
          examples.append(full_sentence)
        elif gold_label == 'stereotype':
          examples.append(full_sentence)
print(examples[:2])
```
## Quantizing Models with Fair-GPTQ

To quantize models using Fair-GPTQ, follow these steps:

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoConfig
# Define models to be quantized
models = ["facebook/opt-13b", "facebook/opt-6.7b"]

# Iterate through each model
for m in models:
    # Define save directory for quantized model
    quant_path = f"{m.split('/')[1]}-int4-fair-gptq"

    # Configure quantization parameters
    quant_config = QuantizeConfig(bits=4, group_size=128, alpha=0.1, desc_act=False, ignore_bias="k_proj,q_proj,v_proj,fc1")
    # Create a directory for logs (if needed)
    !mkdir -p logs
    # Load original model and apply quantization
    model = GPTQModel.load(m, quant_config)
    model.quantize(examples, batch_size=2)
    model.save(quant_path)

    # Load the quantized model from the saved path
    model = GPTQModel.load(quant_path)

    # Example of model generation with the quantized model
    result = model.generate("Uncovering deep insights begins with")[0]  # Example text for generation
    print(model.tokenizer.decode(result))  # Print the generated text

    # Define repository ID and commit message for pushing to hub (replace USERNAME with your username)
    repo_id = f"USERNAME/{quant_path}"
    commit_message = f"Quantized model for {m}"

    # Push the quantized model to a hub (assuming a function like 'push_to_hub' exists in GPTQModel)
    GPTQModel.push_to_hub(repo_id=repo_id, quantized_path=quant_path, exists_ok=True, private=False)
```

## Quantization configurations
In the paper, we use several configurations for two model types: OPT and Mistral.  To use these configurations, we use the following parameters:

1) OPT model family
 - $Fair-GPTQ_{\text{ALL}}$ # all layers
   ```
   quant_config = QuantizeConfig(bits=4, group_size=128, alpha=0.1, desc_act=False, ignore_bias="k_proj,q_proj,v_proj,fc1")
   ```
- $Fair-GPTQ_{\text{l}}$ # lower layers
  ```
  last_layer = int(AutoConfig.from_pretrained(m).num_hidden_layers)
  rounded_down = math.ceil(last_layer * 10 / 100)
  if rounded_down%2!=0:
     rounded_down+=1
  lower_strategy = list(range(last_layer))[:rounded_down]
  selected_strategy = lower_strategy
  result_str = ",".join(map(str, lower_strategy))
  quant_config = QuantizeConfig(bits=4, group_size=128,
                                  alpha=0.1,
                                  select_layer=result_str,
                                  desc_act=False,
                                  ignore_bias="k_proj,q_proj,v_proj,fc1")
   ```
- $Fair-GPTQ_{\text{u}}$ # upper layers
  ```
  last_layer = int(AutoConfig.from_pretrained(m).num_hidden_layers)
  rounded_down = math.ceil(last_layer * 10 / 100)
  if rounded_down%2!=0:
     rounded_down+=1
  upper_strategy = list(range(last_layer))[-rounded_down:]
  selected_strategy = upper_strategy
  result_str = ",".join(map(str, lower_strategy))
  quant_config = QuantizeConfig(bits=4, group_size=128,
                                  alpha=0.1,
                                  select_layer=result_str,
                                  desc_act=False,
                                  ignore_bias="k_proj,q_proj,v_proj,fc1")
   ```
- $Fair-GPTQ_{\text{lu}}$ # lower-upper layers
  ```
  last_layer = int(AutoConfig.from_pretrained(m).num_hidden_layers)
  rounded_down = math.ceil(last_layer * 10 / 100)
  if rounded_down%2!=0:
     rounded_down+=1
  rounded_down=int(rounded_down/2)
  lower_strategy = list(range(last_layer))[:rounded_down]
  upper_strategy = list(range(last_layer))[-rounded_down:]
  selected_strategy = lower_strategy + upper_strategy
  result_str = ",".join(map(str, lower_strategy))
  quant_config = QuantizeConfig(bits=4, group_size=128,
                                  alpha=0.1,
                                  select_layer=result_str,
                                  desc_act=False,
                                  ignore_bias="k_proj,q_proj,v_proj,fc1")
   ```
2) Mistral model   
   

   For Mistral model, same code can be used for different configurations with `ignore_bias="k_proj,q_proj,v_proj,up_proj,gate_proj"`
   
## Configuration file details
```yaml
alpha: float = field(default=0.0) - Alpha used for computing the Hessian H01.
ignore_bias: str = field(default="") - A string listing the types of matrices to be ignored for applying Fair-GPTQ, separated by commas.
For 'opt', it can be: "k_proj,q_proj,v_proj,fc1,out_proj,fc2".
For 'mistral', it can be: "k_proj,q_proj,v_proj,up_proj,o_proj,gate_proj,down_proj".

select_layer: str = field(default="") - A string specifying layers where Fair-GPTQ should be applied, separated by commas. For example, "0,1,2,3".
sum_hessians: int = field(default=1) - Whether to consider the second order in the Hessian.
log_delta_w: int = field(default=0) - Whether to log delta w relative change (Figure 4) in the paper.
log_hessian_sensitivity: int = field(default=0) - Whether to log column-level sensitivities measured with the Hessian-based score. The higher the value, the more column contributes to bias term.

```


# Model Evaluation

For evaluation scripts and results, please refer to the `evaluation/EVALUATION.md`.

# Sanity-Check Experiments 

To confirm that debiasing is specifically due to our proposed modification and the use of paired stereo- and anti-stereotype data, we conducted a sanity check using unrelated pairs from StereoSet. The results showed no bias reduction in this scenario, despite a drop in performance.
For details and code related to these experiments, refer to the `./sanity-check-experiments/` directory.


