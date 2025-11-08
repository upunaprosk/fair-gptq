# Sanity Check Experiments

In this directory, we provide code for the experiments described in §5.1 and Appendix D.

## Unrelated Pairs

We construct unrelated pairs using StereoSet. For this purpose, we prompt the Mistral model to generate unrelated paraphrases. We rely on the Ollama framework ([Ollama](https://github.com/ollama/ollama)) for this task. Code for generating these unrelated pairs can be found in the `ollama-make-unrelated.py` script. 

As a result, we obtain unrelated pairs for StereoSet sentences, such as:
```
BLANK: Context: Every male acts BLANK.
s='Every male acts blue.'
s='Every male acts up.'
```

These pairs are used as calibration data during quantization, and we evaluate the resulting models using them.
We provide a json `ollama-extended.json` with the resulting dict of unrelated pairs. 

## Shuffled Pairs (Random)

As a random baseline, we create shuffled pairs of stereotypical and anti-stereotypical sentences that are not directly related to each other. For instance, 'Ethiopian men are fat' (anti-stereotype) and 'Russian men are rude' (stereotype). We shuffle anti-stereotypical pairs and combine them as follows:

```python
import random
import numpy as np
import torch
import json

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def shuffle_one_group(data, seed: int = 42, shuffle_target: str = "stereotype"):
    """
    Shuffle only one group of StereoSet sentences (stereotype or anti-stereotype),
    using a fixed seed for reproducibility.

    Args:
        data: StereoSet JSON data
        seed: random seed
        shuffle_target: "stereotype" or "anti-stereotype"
    """
    seed_everything(seed)

    examples_stereo = []
    examples_antistereo = []

    for entry in data['data']['intrasentence']:
        for sentence_entry in entry['sentences']:
            sentence = sentence_entry['sentence']
            label = sentence_entry['gold_label']
            if label == 'stereotype':
                examples_stereo.append(sentence)
            elif label == 'anti-stereotype':
                examples_antistereo.append(sentence)

    if shuffle_target == "stereotype":
        random.shuffle(examples_stereo)
    elif shuffle_target == "anti-stereotype":
        random.shuffle(examples_antistereo)
    else:
        raise ValueError("shuffle_target must be 'stereotype' or 'anti-stereotype'")

    return examples_stereo, examples_antistereo

# Example usage:
with open("ollama-extended.1.json") as f:
    data = json.load(f)

stereo, antistereo = shuffle_one_group(data, seed=42, shuffle_target="stereotype")
examples = []
for i in range(len(stereo)):
    examples.append(stereo[i])
    examples.append(antistereo[i])
# examples further to be used in .quantize
```
