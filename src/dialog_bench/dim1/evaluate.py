# %%
import pandas as pd
import ast
import random
import json
from tqdm import tqdm
import warnings
import jsonlines
from dialog_bench.utils import *
tqdm.pandas() 
from pathlib import Path
import numpy as np
# %%
BASE_DIR = Path(__file__).resolve().parents[3] 
HERE = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "processed_datasets.csv"
results_path = BASE_DIR / "results" / "runs.jsonl"
prompt_path = HERE /"prompts.json"
label_path = HERE /"label_map.json"

# %%
def evaluate(data,col,generate,sim_model,nb_samples =100):
    """
    Evaluate the LLM's ability to identify the learning context of a dialogue based on the generated anonymized dialogue.
    """
    with open(label_path, "r") as f:
        label_map = json.load(f)
    with open(prompt_path, "r") as f:
        prompts = json.load(f)
    
    testset = []
    for value in data[col].unique().tolist():
        if value == "None" or value == "" or pd.isna(value):
            continue
        if data[data[col]==value].shape[0] > nb_samples:
            test = data[data[col]==value]
            test["dialogue"]= test.apply(lambda x: generate_anonymized_dialogue(x,chunked=True,max_chunk_size=20,min_chunk_size=5)[0],axis=1)
            test = test[test["dialogue"].apply(lambda x: len(x) != 0)]
            test = test.sample(min(nb_samples,test.shape[0]))
            
            testset.append(test[["dialogue",col]])
        else:
            examples = []
            test = data[data[col]==value]
            for ind,row in test.iterrows():
                dialogues = generate_anonymized_dialogue(row,chunked=True,max_chunk_size=20)
                for dialogue in dialogues:
                    examples.append({"dialogue":dialogue,col:row[col]})

            testset.append(pd.DataFrame(examples).sample(min(nb_samples,len(examples))))
    testset = pd.concat(testset,ignore_index=True).sample(frac=1)
    warnings.warn("Generating resposes for the test set, this may take a while...")
    testset["generated"] = testset["dialogue"].progress_apply(lambda x: generate(random.choice(list(prompts[col].values())).format(INPUT_DIALOGUE=x)))
    if col in label_map.keys():
        score = (testset["generated"].apply(lambda x: label_map[col].get(x[0],""))==testset[col]).mean()
        
    else:
        warnings.warn("Not a MCQ column, using cosine similarity for evaluation, this may take a moment...")
        score = (testset.progress_apply(lambda x: compute_similarity(sim_model,x["generated"], x[col]), axis=1)).mean()

    return score





# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LLM abilities")
    parser.add_argument("--column", type=str, help="The column to evaluate")
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
    
    
    
    scores = []
    if args.column:
        columns = [x.strip() for x in args.column.split(",")]
        for col in columns:
            if not col in ["learning_context", "comm_modality","agent_config", "subject", "edu_level"]:
                warnings.warn(f"Feature {col} not recognized. Skipping ...")
                continue
            else: 
                score = evaluate(data,args.column,generate,sim_model,args.samples)
                with jsonlines.open(results_path, mode='a') as writer:
                    writer.write({"dim":"1","column":  args.column, "score": score, "model": args.model, "nb_samples":args.samples,"seed":RD,"args":args.gen_args})
                print(f"Score for {args.column}: {score}")
    else:
        columns = ["learning_context", "comm_modality","agent_config", "subject", "edu_level"]
        for col in columns:
            score = evaluate(data,col,generate,sim_model,args.samples)
            with jsonlines.open(results_path, mode='a') as writer:
                writer.write({"dim":"1","column": col, "score": score, "model": args.model, "nb_samples":args.samples,"seed":RD,"args":args.gen_args})
            print(f"Score for {col}: {score}")
