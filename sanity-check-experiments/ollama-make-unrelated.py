import os
import sys
import json
import ollama

# Model name for local LLM via Ollama (ministral-8b-instruct-2410_q4km is the best for now)
MODEL_NAME = "cas/ministral-8b-instruct-2410_q4km" # tested (in the desc. order by quality): "dolphin3" "phi4-mini:3.8b" "llama3.2:1b" 

# System prompt remains the same
system_prompt = (
    "You are a helpful assistant that generates syntactically valid but semantically nonsensical sentences "
    "based on a given context sentence with a BLANK token. Fill the BLANK with a grammatically correct word, "
    "but make the sentence unrelated or nonsensical. Avoid using color names for placeholder, it's not very useful. "
    "Do not leave BLANK placeholder unchanged!"
)

def load_stereoset_json():
    path = os.path.join(os.environ['HOME'], '.cache', 'stereoset', 'dev.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def add_dot(s):
    if not s.endswith('.') and not s.endswith('?') and not s.endswith('!'):
        s += '.'
    return s

def unquote(s):
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
    return s

def check_sentence_by_context(context, sentence):
    "Check if sentence is the context with BLANK placeholder substituted by exactly one word"
    sentence = sentence.strip()
    context = context.strip()
    if not sentence[0].isupper():
        sentence = sentence[0].upper() + sentence[1:]
    if not context[0].isupper():
        context = context[0].upper() + context[1:]
    sentence = unquote(sentence)
    sentence = add_dot(sentence)
    context = add_dot(context)
    context_words = context.split()
    sentence_words = sentence.split()
    if len(sentence_words) != len(context_words):
        print(f"{context=}", flush=True)
        return False
    for i, w in enumerate(context_words):
        if w.lower() != sentence_words[i].lower():
            # allow only BLANK => <one_word> substitution
            if 'BLANK' not in w:
                print(f"{context=}", flush=True)
                return False
    return True

def generate_sentences(context=None, stereotype=None, anti_stereotype=None, unrelated=None, n_sentences=2):
    # Build messages using Ollama's format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}\nExample sentence: {stereotype}"},
        {"role": "assistant", "content": "Stereotype example."},
        {"role": "user", "content": f"Context: {context}\nExample sentence: {anti_stereotype}"},
        {"role": "assistant", "content": "Anti-stereotype example."},
        {"role": "user", "content": f"Context: {context}\nExample sentence: {unrelated}"},
        {"role": "assistant", "content": "Unrelated but syntactically valid example."},
        {"role": "user", "content": f"Context: {context}\nGenerate a new unrelated sentence with correct syntax. Use only one word for the BLANK placeholder. Note that unrelated sentence should not resemble neither stereotype nor anti-stereotype sentence! However, use only the correct part of speech for BLANK placeholder even if the given above example of unrelated sentence was wrong. Never leave BLANK placeholder in your answer."} # this causes the answers to be not unrelated:  Check twice the part of speech that you use, in doubt use another word!
    ]

    sentences = []
    cnt = 0
    max_tries = 200
    while len(sentences) < n_sentences and cnt <= max_tries:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0.9,
                "num_predict": 100,  # Ollama equivalent to max_tokens
                "stop": ["\n"]
            }
        )
        generated_text = response['message']['content']
        sentence = generated_text.strip()
        cnt += 1
        if cnt > max_tries:
            print(f"GIVE UP: {context}", flush=True)
            break
        if 'BLANK' in sentence:
            print(f"BLANK: {sentence}", flush=True)
            continue
        sentence = sentence.replace('*', '')
        sentence = sentence.replace('_', '')
        sentence = add_dot(sentence)
        if sentence.startswith('Example: '):
            sentence = sentence.replace('Example: ', '')
        if sentence in sentences:
            print(f"REPEAT: {sentence}", flush=True)
            continue
        if not check_sentence_by_context(context, sentence):
            print(f"MALFORMED: {sentence}", flush=True)
            continue
        sentences.append(sentence)
    if len(sentences) < n_sentences:
        return []
    return sentences

# process_and_extend() remains unchanged
def process_and_extend(data):
    limit = 0
    count = 0
    for entry in data['data']['intrasentence']:
        stereo = None
        anti = None
        unrelated = None
        for sentence_entry in entry['sentences']:
            sentence = sentence_entry['sentence']
            gold_label = sentence_entry['gold_label']
            if gold_label == 'anti-stereotype':
                anti = sentence
            elif gold_label == 'stereotype':
                stereo = sentence
            elif gold_label == 'unrelated':
                unrelated = sentence
        print("")
        for i, s in enumerate(generate_sentences(context=entry['context'], 
                                                stereotype=stereo, 
                                                anti_stereotype=anti, 
                                                unrelated=unrelated)):
            print(f"{s=}", flush=True)
            entry['sentences'].append({'sentence': s, 'gold_label': f"unrelated{i+1}"})
        count += 1
        if limit > 0 and count > limit:
            break

if __name__ == "__main__":
    stereoset_data = load_stereoset_json()
    process_and_extend(stereoset_data)
    with open("ollama-extended.json", "wt") as f:
        print(json.dumps(stereoset_data, sort_keys=True, indent=4), file=f)
