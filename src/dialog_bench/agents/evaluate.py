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
import importlib.util
import inspect

# %%
BASE_DIR = Path(__file__).resolve().parents[3] 
HERE = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "processed_datasets.csv"
results_path = BASE_DIR / "results" / "runs.jsonl"
prompt_path = BASE_DIR /"src" / "dialog_bench" / "dim3" /"prompts.json"
taxonomy_path = BASE_DIR /"src" / "dialog_bench" / "dim3" /"taxonomies.json"


SYSTEM = """
<|im_start|>system
You are a dialogic act annotation model.

Your task is to classify the next dialogue turn by selecting exactly one dialogic act from the available tools.

Each tool represents one possible dialogic act label:
- The tool name is the label.
- The tool description defines when that label applies.

You will be given:
- An annotated dialogue chunk, where each previous turn is already labeled.
- A next turn whose label is missing.

Use the annotated dialogue chunk as context and select the single most appropriate dialogic act for the next turn.

You must call one or more tools.

Do not provide explanations, reasoning, confidence scores, comments, or any text outside the tool call.

If none of the dialogic act labels applies, call the None tool.

Your entire response must be the tool calls.

{TOOLS}
<|im_end|>\n
<|im_start|> user
Annotated dialogue chunk:
{ANNOTATED_DIALOGUE}

Next turn:
{NEXT_TURN}

Select the dialogic act for the next turn.
<|im_end|>\n
<|im_start|> assistant\n
<think>\n\n</think>\n\n"""


def create_tool(pair):
    
    return json.dumps(
    {
    "type": "function",
    "function": {
      "name": pair[0],
      "description": pair[1]}
    })

def create_tools(dataset):

    with open(taxonomy_path,"r") as f:
        taxonomy = json.load(f).get(dataset,"")
    tools = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
    for pair in taxonomy.items():
        tools += "\n" + create_tool(pair)
    tools += "\n</tools>"
    tools +=  '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags. You could have multiple functions inside tool call tags.</IMPORTANT>' 
    return tools

def turn_to_history(turn, acts=None, step=None):
    speaker = turn.split(":",1)[0]
    utterance = turn.split(":",1)[1]
    if acts!= None:
        tools = "\n".join([f"<function={act}></function>" for act in acts.split(",")])
        return f"<turn {step}, speaker={speaker}> {utterance}</turn>\n<tool_call>\n{tools}\n</tool_call>"
    else:
        return f"<turn {step}, speaker={speaker}> {utterance}</turn>"

def evaluate_annotation(data,generate,nb_samples,task="act_prediction"):
    with open(prompt_path,"r") as f:
        prompts = json.load(f)
    with open(taxonomy_path,"r") as f:
        taxonomies = json.load(f)
    multi = ["ncte","tscc","weighttasks"]

    tools = {}
    for dataset in taxonomies.keys():
        
        tools[dataset] = create_tools(dataset)


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

                    next_turn = turn_to_history(dialogue,None,0)
                    next_act = act_list[-1]
                else:
                    past_dialogue = dialogue.rsplit("\n",1)[0]
                    next_turn = turn_to_history(dialogue.rsplit("\n",1)[1],None,len(act_list)-1)
                    next_act = act_list[-1]
                    annotated_dialogue = "\n".join([turn_to_history(turn,act_list[i],i) for i,turn in enumerate(past_dialogue.split("\n"))])
                taxonomy = tools[row["dataset"]]
                df.append({"dialogue":annotated_dialogue, "turn":next_turn, "act":next_act.lower(),"taxonomy":taxonomy,"dataset":row["dataset"]})
        else:
            speakers = ast.literal_eval(row["speakers"]) if isinstance(row["speakers"], str) else row["speakers"]
            turns = ast.literal_eval(row["turns"]) if isinstance(row["turns"], str) else row["turns"]
            current_acts =  ast.literal_eval(row["dialogic_acts"]) if isinstance(row["dialogic_acts"], str) else row["dialogic_acts"]
            annotated_dialogue = ""
            for step, (speaker,turn,act) in enumerate(zip(speakers,turns,current_acts)):
                utterance = speaker+": "+turn
                if speaker!="Teacher":
                    annotated_dialogue+= "\n" + turn_to_history(utterance,None,step)
                else:
                    taxonomy =tools[row["dataset"]]
                    df.append({"dialogue":annotated_dialogue, "turn":turn_to_history(utterance,None,step), "act":act.lower(),"taxonomy":taxonomy,"dataset":row["dataset"]})
                    annotated_dialogue += "\n" +  turn_to_history(utterance,act.lower(),step)

    df = pd.DataFrame(df)
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
    testset["generated"] = testset.progress_apply(lambda x: generate(SYSTEM.format(TOOLS = x["taxonomy"],ANNOTATED_DIALOGUE=x["dialogue"],NEXT_TURN=x["turn"])).lower(),axis=1)
    testset["score"] = example_f1(testset,taxonomies)
    print(testset.groupby("dataset")["score"].mean())
    testset.to_csv(str(HERE) +"/test.csv")
    
    return testset["score"].mean()


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
        generate = lambda message: generate_hf(message,model,tokenizer,args.type,ast.literal_eval(args.gen_args),ignore_template=True)
    else:   
        raise NotImplementedError("Backend is not supported for now")
    
    score = evaluate_annotation(data,generate,args.samples,"act_annotation")
    with jsonlines.open(results_path, mode='a') as writer:
        writer.write({"dim":"3","column": "acts_annotation", "score": score, "model": args.model, "nb_samples":args.samples,"seed":RD,"args":args.gen_args})

