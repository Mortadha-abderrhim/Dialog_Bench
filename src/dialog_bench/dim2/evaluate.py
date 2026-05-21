
import pandas as pd
import ast
import random
import json
from tqdm import tqdm
import warnings
import jsonlines
from dialog_bench.utils import *
import numpy as np
tqdm.pandas() 

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3] 
HERE = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "processed_datasets.csv"
results_path = BASE_DIR / "results" / "runs.jsonl"
prompt_path = HERE /"prompts.json"


def evaluate_teacher(data,generate,nb_samples=100):
    
    with open(prompt_path,"r") as f:
        prompts = list(json.load(f)["teacher_identification"].values())
    data = data[data["speakers"].apply(lambda speakers: "tutor" in set([str(x).lower() for x in speakers]) or "teacher" in set([str(x).lower() for x in speakers]) or "t" in set([str(x).lower() for x in speakers]))]
    data["teacher"] = data["speaker_mapping"].apply(lambda x: x.get("teacher",x.get("Teacher",x.get("tutor",x.get("Tutor",x.get("t",x.get("T",None)))))))
    data["dialogue"] = data.apply(lambda x: random.choice([dialog if x["teacher"] in dialog else "" for dialog in generate_anonymized_dialogue(x,chunked=True,max_chunk_size=min(len(x["speakers"]),30)) ]),axis=1)
    data = data[data.apply(lambda x: x["teacher"] in x["dialogue"],axis=1)]
    testset = data[["dialogue","teacher"]].sample(nb_samples)
    testset["generated"] = testset["dialogue"].progress_apply(lambda x: generate(random.choice(prompts).format(INPUT_DIALOGUE=x)))
    score = (testset["generated"]==testset["teacher"]).mean()
    return score

def evaluate_next_speaker(data,generate,nb_samples = 100):
    with open(prompt_path,"r") as f:
        prompts = list(json.load(f)["speaker_identification"].values())
    data["dialogue"] = data.apply(lambda x: generate_anonymized_dialogue(x,chunked=True,max_chunk_size=min(len(x["speakers"]),30)),axis=1)
    df = []
    for chunks in data["dialogue"]:
        for chunk in chunks:
            if len(chunk.strip())==0:
                continue
            try:
                past = chunk.rsplit("\n",1)[0]
                next = chunk.rsplit("\n",1)[1].split(":",1)[1]
                speaker = chunk.rsplit("\n",1)[1].split(":",1)[0]
            except:
                print(chunk.rsplit("\n",1))
                raise KeyError
            if speaker in past:
                df.append({"past":past,"next":next,"speaker":speaker})
    testset = pd.DataFrame(df).sample(nb_samples)
    testset["generated"] = testset.progress_apply(lambda x: generate(random.choice(prompts).format(INPUT_DIALOGUE=x["past"],NEXT_TURN=x["next"])),axis=1)
    score = (testset["generated"]==testset["speaker"]).mean()
    return score

def evaluate_turn(data,generate,sim_model,pos = "next",nb_samples=100):
    with open(prompt_path,"r") as f:
        prompts = list(json.load(f)[f"{pos}_turn"].values())

    data["chunks"] = data.apply(lambda x: [random.choice(generate_anonymized_dialogue(x,chunked=True,max_chunk_size=min(10,len(x["speakers"]))))],axis=1)
    data = data[data["chunks"].apply(lambda x: x!=[""])].sample(int(nb_samples*1.5))
    df = []
    warnings.warn("Creating the task ...")
    for ind,row in tqdm(data.iterrows()):
        turns = ast.literal_eval(row["turns"]) if isinstance(row["turns"], str) else row["turns"]
        turns = [str(x) for x in turns]
        turn_embeddings = embed(sim_model,turns)
        for chunk in row["chunks"]:
            if pos == "next":
                dialogue = chunk.rsplit("\n",1)[0]
                correct = chunk.rsplit("\n",1)[1].split(":",1)[1]
            else:
                dialogue = chunk.split("\n",1)[1]
                correct = chunk.split("\n",1)[0].split(":",1)[1]
            correct_embedding = embed(sim_model,correct)
            similarities = sim_model.similarity(turn_embeddings, correct_embedding).numpy().ravel()
            indices_desc = np.argsort(similarities)[::-1]
            choices = []
            i=1
            while len(choices)<3 and i<len(turns):
                if not turns[indices_desc[i]].replace("\n"," ") in dialogue and similarities[indices_desc[i]] < 0.9:
                    choices.append(turns[indices_desc[i]].replace("\n"," "))
                i+=1
            if i>=len(turns) or len(choices)<3:
                continue
            choices.append(correct)
            random.shuffle(choices)
            correct = "ABCD"[choices.index(correct)]
            df.append({"dialogue":dialogue,"correct":correct,"options":choices})
    testset = pd.DataFrame(df).sample(nb_samples)
    warnings.warn("Generating Model Answers ...")
    testset["generated"] = testset.progress_apply(lambda x: generate(random.choice(prompts).format(INPUT_DIALOGUE=x["dialogue"],OPTION_A=x["options"][0],OPTION_B=x["options"][1],OPTION_C=x["options"][2],OPTION_D=x["options"][3])),axis=1)
    score = (testset["generated"]==testset["correct"]).mean()
    return score

def evaluate_chunk_ordering(data,generate,sim_model,nb_samples = 100):
    with open(prompt_path,"r") as f:
        prompts = list(json.load(f)["chunk_ordering"].values())
    data["dialogue"] = data.apply(lambda x: generate_anonymized_dialogue(x,chunked=True,max_chunk_size= min(10,len(x["speakers"])//2 -1)),axis=1)
    data = data[data["dialogue"].apply(lambda x: len(x)>2)].sample(nb_samples)
    df = []
    warnings.warn("Creating the task ...")
    for chunks in tqdm(data["dialogue"]):
        if chunks == 3:
            fst = chunks[0]
            choices = [chunks[0],chunks[2]]
            random.shuffle(choices)
            df.append({"chunks":choices,"correct":"AB"[choices.index(fst)]})
        else:
            embeddings = embed(sim_model,chunks)
            similarities = sim_model.similarity(embeddings,embeddings).numpy()
            row_indices, col_indices = np.triu_indices_from(similarities, k=1)
            upper_triangle_values = similarities[row_indices, col_indices]
            
            sorted_indices = np.argsort(upper_triangle_values)[::-1]

            top_row_indices = None
            top_col_indices = None

            for idx in sorted_indices:
                row = row_indices[idx]
                col = col_indices[idx]

                # Keep only non-consecutive indices
                if abs(row - col) != 1:
                    top_row_indices = row
                    top_col_indices = col
                    break
            fst = chunks[min(top_row_indices,top_col_indices)]
            choices = [fst,chunks[max(top_row_indices,top_col_indices)]]
            random.shuffle(choices)
            df.append({"chunks":choices,"correct":"AB"[choices.index(fst)]})
    testset = pd.DataFrame(df).sample(frac=1.0)
    warnings.warn("Generating model answers ...")
    testset["generated"] = testset.progress_apply(lambda x: generate(random.choice(prompts).format(CHUNK_A=x["chunks"][0],CHUNK_B=x["chunks"][1])),axis=1)
    score = (testset["generated"]==testset["correct"]).mean()
    return score




    






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LLM abilities")
    parser.add_argument("--backend", type=str, help="The backend to use for evaluation")
    parser.add_argument("--url", type=str, help="The URL for ollama", default=None)
    parser.add_argument("--model", type=str, help="The model name ", default=None)
    parser.add_argument("--type", type=str, help="The type to use for unsloth", default="text")
    parser.add_argument("--gen_args", type=str, help="Generation arguments for unsloth in dict format", default="{}")
    parser.add_argument("--random", type=int, help="Generation arguments for unsloth in dict format", default=42)
    parser.add_argument("--samples", type=int, help="Generation arguments for unsloth in dict format", default=100)
    args = parser.parse_args()

    RD = args.random
    random.seed(RD)
    np.random.seed(RD)
    sim_model = load_similarity()
    data = pd.read_csv(data_path)
    data["speakers"] = data["speakers"].apply(lambda speakers: ast.literal_eval(speakers) if isinstance(speakers, str) else speakers)
    data["speaker_mapping"] = data["speakers"].apply(lambda x: anonymize_speakers(x))

    if args.backend == "ollama":
        generate = lambda message: generate_ollama(message,args.url,args.model,ast.literal_eval(args.gen_args).update({"seed":RD}))
    elif args.backend == "openai":
        generate = lambda message: generate_openai(message,args.model)
    elif args.backend == "unsloth":
        model, tokenizer = load_unsloth(args.model,args.type)
        generate = lambda message: generate_unsloth(message,model,tokenizer,args.type,ast.literal_eval(args.gen_args))
    elif args.backend == "hf":
        model, tokenizer = load_model_hf(args.model,args.type)
        generate = lambda message: generate_hf(message,model,tokenizer,args.type,ast.literal_eval(args.gen_args))
    else:   
        raise NotImplementedError("Backend is not supported for now")
    
    
    score = evaluate_teacher(data,generate,args.samples)
    with jsonlines.open(results_path, mode='a') as writer:
        writer.write({"dim":"2","column": "teacher_id", "score": score, "model": args.model, "nb_samples":args.samples,"seed":RD,"args":args.gen_args})
    score = evaluate_next_speaker(data,generate,args.samples)
    with jsonlines.open(results_path, mode='a') as writer:
        writer.write({"dim":"2","column": "speaker_id", "score": score, "model": args.model, "nb_samples":args.samples,"seed":RD,"args":args.gen_args})
    score = evaluate_turn(data,generate,sim_model,"next",args.samples)
    with jsonlines.open(results_path, mode='a') as writer:
        writer.write({"dim":"2","column": "next_turn", "score": score, "model": args.model, "nb_samples":args.samples,"seed":RD,"args":args.gen_args})
    score = evaluate_turn(data,generate,sim_model,"previous",args.samples)
    with jsonlines.open(results_path, mode='a') as writer:
        writer.write({"dim":"2","column": "previous_turn", "score": score, "model": args.model, "nb_samples":args.samples,"seed":RD,"args":args.gen_args})
    score = evaluate_chunk_ordering(data,generate,sim_model,args.samples)
    with jsonlines.open(results_path, mode='a') as writer:
        writer.write({"dim":"2","column": "chunk_ordering", "score": score, "model": args.model, "nb_samples":args.samples,"seed":RD,"args":args.gen_args})

    