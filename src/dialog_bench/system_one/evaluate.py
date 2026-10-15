import json

from huggingface_hub import hf_hub_download
from transformers import AutoModel


from dialog_bench.utils import *

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
taxonomy_path = HERE /"taxonomies.json"




def format_dialogues(data,task,nb_samples):
    with open(taxonomy_path,"r") as f:
        taxonomies = json.load(f)

    multi = ["ncte","tscc","weighttasks"]

    testset = data[data["dataset"].apply(lambda x: x in list(taxonomies.keys()))]
    df = []
    for ind,row in testset.iterrows():
        if row["dataset"] != "mathdial":
            dialogues, acts = generate_anonymized_dialogue(row,chunked=True,max_chunk_size=10,min_chunk_size=1,return_acts=True)
            for dialogue, act_list in zip(dialogues,acts):
                if (len(set(act_list)) ==1 and act_list[0].lower() == "none") or len(dialogue)==0:
                    continue

                if len(act_list)==1:
                    if(task=="act_prediction"):
                        ### No prediction if no past dialogue
                        continue
                    annotated_dialogue = ""
                    next_turn = dialogue
                    next_act = act_list[-1]
                else:
                    past_dialogue = dialogue.rsplit("\n",1)[0]
                    next_turn = dialogue.rsplit("\n",1)[1]
                    next_act = act_list[-1]
                    annotated_dialogue = "\n".join([turn+ " <act = " + act_list[i].lower()+">" for i,turn in enumerate(past_dialogue.split("\n"))])
                taxonomy =taxonomies[row["dataset"]]
                context = "Educational level: "+str(row["edu_level"]) + "\nLearning context: "+ str(row["learning_context"]) + "\nSubject: "+ str(row["subject"]) + "\nCommunication modality: "+ str(row["comm_modality"]) + "\nAgents configuration: "+ str(row["agent_config"])
                df.append({"dialogue":annotated_dialogue, "turn":next_turn, "act":next_act.lower(),"taxonomy":taxonomy,"dataset":row["dataset"],"context":context})
        else:
            speakers = ast.literal_eval(row["speakers"]) if isinstance(row["speakers"], str) else row["speakers"]
            turns = ast.literal_eval(row["turns"]) if isinstance(row["turns"], str) else row["turns"]
            current_acts =  ast.literal_eval(row["dialogic_acts"]) if isinstance(row["dialogic_acts"], str) else row["dialogic_acts"]
            annotated_dialogue = ""
            for speaker,turn,act in zip(speakers,turns,current_acts):
                if speaker!="Teacher":
                    annotated_dialogue+= speaker + ": " + turn +"\n"
                else:
                    taxonomy =taxonomies[row["dataset"]]
                    context = "Educational level: "+str(row["edu_level"]) + "\nLearning context: "+ str(row["learning_context"]) + "\nSubject: "+ str(row["subject"]) + "\nCommunication modality: "+ str(row["comm_modality"]) + "\nAgents configuration: "+ str(row["agent_config"])

                    df.append({"dialogue":annotated_dialogue, "turn":turn, "act":act.lower(),"taxonomy":taxonomy,"dataset":row["dataset"],"context":context})
                    annotated_dialogue += speaker + ": " + turn +"<act = "+ act.lower() +">" +"\n"
        
                    
    df = pd.DataFrame(df)
    ## We choose to remove examples where the act to be predicted is not in the taxonmy usually due to typos in the data (can be matched to closest but ...)
    df = df[df.apply(lambda x: all([y.strip() in [z.lower() for z in list(taxonomies[x["dataset"]].keys())]for y in x["act"].split(",")]),axis=1)]


    testset = []
    for tax, grp in df.groupby("dataset"):

        if "none" in grp["act"].unique():
            nones = grp[grp["act"]=="none"].sample(int(min(grp[grp["act"]=="none"].shape[0],nb_samples*0.1)))
            testset.append(nones)
            not_nones = grp[grp["act"]!="none"].sample(int(min(grp[grp["act"]!="none"].shape[0],nb_samples-nones.shape[0])))
            testset.append(not_nones)
        else:
            testset.append(grp.sample(min(grp.shape[0],nb_samples)))
    testset = pd.concat(testset,ignore_index=True)
    return testset


# Text
def annotate(model,x):
    result = model.system_one(
        state="Dialogue:\n" + x["dialogue"] + "\nNext turn:\n" + x["turn"],
        questions={
            "dialogic_act": {
                "type": "choice",
                "instructions": "Which dialogic act among the following is the most appropriate for the next turn in the dialogue?",
                "criteria": x["taxonomy"],
            },
        },
    )
    res = json.dumps(result["answers"]["dialogic_act"]["choice"], indent=2)
    print(res)
    return res

def predict(model,x):
    result = model.system_one(
        state="Dialogue:\n" + x["dialogue"],
        questions={
            "dialogic_act": {
                "type": "choice",
                "instructions": "Which dialogic act among the following is the most appropriate for the next turn in the dialogue?",
                "criteria": x["taxonomy"],
            },
        },
    )
    res = json.dumps(result["answers"]["dialogic_act"]["choice"], indent=2)
    print(res)
    return res

def run_task(data,task,nb_samples,model):
    with open(taxonomy_path,"r") as f:
        taxonomies = json.load(f)
    testset = format_dialogues(data,task,nb_samples)
    if task=="act_prediction":
        testset["generated"] = testset.progress_apply(lambda x: predict(model, x), axis=1)
    else:
        testset["generated"] = testset.progress_apply(lambda x: annotate(model, x), axis=1)
    
    testset["score"] = example_f1(testset,taxonomies)
    return testset["score"].mean(),testset.groupby("dataset")["score"].mean().to_dict()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LLM abilities")
    parser.add_argument("--model", type=str, help="The model name ", default=None)
    parser.add_argument("--random", type=int, help="Generation arguments for unsloth in dict format", default=42)
    parser.add_argument("--samples", type=int, help="Generation arguments for unsloth in dict format", default=100)

    args = parser.parse_args()
    
    RD = args.random
    random.seed(RD)
    np.random.seed(RD)

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)

    data = pd.read_csv(data_path)
    data["speakers"] = data["speakers"].apply(lambda speakers: ast.literal_eval(speakers) if isinstance(speakers, str) else speakers)
    data["speaker_mapping"] = data["speakers"].apply(lambda x: anonymize_speakers(x))

    score,detailed = run_task(data,"act_annotation",args.samples,model)
    with jsonlines.open(results_path, mode='a') as writer:
       writer.write({"dim":"system_one","column": "acts_annotation", "score": score, "model": args.model, "nb_samples":args.samples,"seed":RD,"detailed":detailed})
    score,detailed = run_task(data,"act_prediction",args.samples,model)
    with jsonlines.open(results_path, mode='a') as writer:
       writer.write({"dim":"system_one","column": "acts_prediction", "score": score, "model": args.model, "nb_samples":args.samples,"seed":RD,"detailed":detailed})