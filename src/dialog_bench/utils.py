# %%
import ast
import random
import string
from sentence_transformers import SentenceTransformer
import requests
from tqdm import tqdm
from openai import OpenAI
import os
import numpy as np
tqdm.pandas() 



# %%
def load_similarity():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(sim_model, sentences):
    return sim_model.encode(sentences)
def compute_similarity(sim_model, sentence1, sentence2):  
  # Compute embeddings for both lists
    embeddings1 = sim_model.encode(sentence1)
    embeddings2 = sim_model.encode(sentence2)

    # Compute cosine similarities
    similarity= sim_model.similarity(embeddings1, embeddings2)[0]
    return similarity

# %%
def anonymize_speakers(speakers):
    def generate_id():
        letters = ''.join(random.choices(string.ascii_uppercase, k=3))
        numbers = ''.join(random.choices(string.digits, k=2))
        return letters + numbers
    speakers = ast.literal_eval(speakers) if isinstance(speakers, str) else speakers
    unique_speakers = list(set(speakers))
    mapping = {speaker: generate_id() for i,speaker in enumerate(unique_speakers)}
    return mapping


def generate_anonymized_dialogue(row,format = "{speaker}: {turn}" ,chunked=False, max_chunk_size=10, return_acts = False,min_chunk_size=4):
    """
    Generate an anonymized dialogue based on the input row.
    If chunked is True, the dialogue will be split into chunks of a RANDOM size with a minimum of 2 turns and a maximum max_chunk_size turns. Choose a random chunk size for each chunk between 2 and max_chunk_size, so that the chunks are not always the same size.
    Use the max_chunk_size as a starting point to get the following chunk and return a list of chunks containing len(turns) // max_chunk_size chunks. If the last chunk has less than 2 turns, merge it with the previous chunk.
    Every chunk should be a string of turns formatted based on the input format, where {speaker} is replaced with the anonymized speaker ID and {turn} is replaced with the turn text followed by a newline character.
    """
    speakers = ast.literal_eval(row["speakers"]) if isinstance(row["speakers"], str) else row["speakers"]
    try:
        turns = ast.literal_eval(row["turns"]) if isinstance(row["turns"], str) else row["turns"]
    except:
        if return_acts:
            return [""],["None"]
        return [""]
    
    
    if return_acts:
        acts =  ast.literal_eval(row["dialogic_acts"]) if isinstance(row["dialogic_acts"], str) else row["dialogic_acts"]
    speaker_mapping = row["speaker_mapping"]
    dialogue = []
    for speaker, turn in zip(speakers, turns):
        anonymized_speaker = speaker_mapping[speaker]
        dialogue.append(format.format(speaker=anonymized_speaker, turn=turn))
    dialogue = "\n".join(dialogue)


    if chunked:
        chunks = []
        if return_acts:
            ret_acts= []
        i = 0
        while i < len(turns):
            chunk_size = random.randint(min(min_chunk_size,max_chunk_size), max_chunk_size)
            chunk_turns = turns[i:i+chunk_size]
            chunk_speakers = speakers[i:i+chunk_size]
            if return_acts:
                chunk_acts = [x if isinstance(x, str) else ", ".join(x) for x in acts[i:i+chunk_size]]
            chunk_dialogue = []
            for speaker, turn in zip(chunk_speakers, chunk_turns):
                anonymized_speaker = speaker_mapping[speaker]
                chunk_dialogue.append(format.format(speaker=anonymized_speaker, turn=str(turn).replace("\n"," ")))
            chunks.append("\n".join(chunk_dialogue))
            if return_acts:
                ret_acts.append(chunk_acts)
            i += chunk_size
        # Merge last chunk if it has less than 2 turns
        if len(chunks) > 1 and len(chunks[-1].split("\n")) < min(min_chunk_size,max_chunk_size):
            chunks[-2] += "\n" + chunks[-1]
            chunks.pop()

        if return_acts:
            return chunks,ret_acts
        return chunks
    return [dialogue]

# %%
def generate_ollama(message,url,model,options={}):
    
    headers = {
    'Authorization': f'Bearer {os.environ.get("OLLAMA_API_KEY")}',
    'Content-Type': 'application/json'
    }
    data = {
    "model": model,
    "messages":[{"role": "user", "content": message}],
    "stream": False,
    "options": options
    }
    response = requests.post(url, headers=headers, json=data)
    response = response.json()["message"]["content"]
    
    return response

"""
    Generate a response using openAI API, the input message is a parameter string, the name of the model as well, take API key from environment
"""
def generate_openai(message,model):
    
    client = OpenAI(os.environ.get("OPENAI_API_KEY"))
    response = client.responses.create(
        model=model,
        input=message
    )
    return response.output_text
def load_unsloth(model,type):
    from unsloth import FastVisionModel,FastLanguageModel
    if type == "vision":
        model, tokenizer = FastVisionModel.from_pretrained(
            model,
            load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        )
        FastVisionModel.for_inference(model)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model,
            load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
        )
        FastLanguageModel.for_inference(model)
    return model, tokenizer
def generate_unsloth(message,model,tokenizer,type,gen_args = {}):
    if type == "text":
        messages = [
            {"role" : "user", "content" : message}
        ]
        tokenized = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors = "pt",return_dict=True).to("cuda")
        output = model.generate(
        **tokenized,
        **gen_args
        )[0]

        response = tokenizer.decode(output[len(tokenized["input_ids"][0]):])

    else:
        messages = [
        {"role" : "user", "content" :message}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        inputs = tokenizer(
            None,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")
        output = model.generate(**inputs, **gen_args)[0]
        response = tokenizer.decode(output[len(tokenized["input_ids"][0]):])

    return response


def example_based_f1(y_true,y_pred):
    f1s= []
    for t,p in list(zip(y_true,y_pred)):
        tp = np.sum((t == 1) & (p == 1))
        true_count = np.sum(t == 1)
        pred_count = np.sum(p == 1)
        f1 = (2 * tp) / (true_count + pred_count+ 1e-9)
        f1s.append(f1)
    return f1s

def example_f1(val,taxonomies):
    tax_map = lambda x: [tax.lower() for tax in list(taxonomies[x["dataset"]].keys())]
    val["classes"] = val.apply(lambda x:tax_map(x),axis=1)
    y_true = val.apply(lambda x: np.array([1 if y in x["act"]  else 0 for y in x["classes"]]),axis=1).to_numpy()
    y_pred = val.apply(lambda x: np.array([1 if y in x["generated"].lower()   else 0 for y in x["classes"]]),axis=1).to_numpy()

    val["scores"] =  example_based_f1(y_true,y_pred)
    return val["scores"]