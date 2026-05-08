# %%
import pandas as pd
import os
import warnings
from Levenshtein import distance
from datasets import load_dataset



# %%
FEATURES = [
    "turns",
    "speakers",
    "dialogic_acts",
    "learning_context",
    "comm_modality",
    "agent_config",
    "subject",
    "edu_level",
    "dataset",
    "language"
]
DATASETS = ["ncte","talkmoves","eedi","tscc","mathdial","weighttasks","delidata"]
CONTEXTS = ["classroom","classroom","tutoring","tutoring","tutoring","collaborative work","collaborative work"]
MODALITIES = ["in-person","in-person","synchronous chat","synchronous chat","synchronous chat","in-person","synchronous chat"]
CONFIGS = ["Human-Human","Human-Human","Human-Human","Human-Human","Human-AI","Human-Human","Human-Human"]
SUBJECTS = ["maths","maths","maths","language","maths","physics, problem solving","problem solving"]
LANGUAGE = ["english","english","english","english","english","english","english"]

def create_mapping_dict():
    mapping_dict = {}
    for dataset, context, modality, config, subject in zip(DATASETS, CONTEXTS, MODALITIES, CONFIGS, SUBJECTS):
        mapping_dict[dataset] = {"learning_context": context,
                                 "comm_modality": modality,
                                 "agent_config": config,
                                 "subject": subject,
                                 "language": LANGUAGE[0],
                                 "dataset": dataset}
    return mapping_dict
metadata = create_mapping_dict()

# %%
def load_ncte(save= True):
    """
    Loads the NCTE dataset, merges the annotations, and returns a dataframe with the standardized format. If the preprocessed file already exists, it loads it directly. Otherwise, it processes the raw files.
    
    Args:
        save (bool): Whether to save the preprocessed dataframe as a CSV file for future use. Default is True.  
    Returns:
        pd.DataFrame: A dataframe containing the NCTE dataset with standardized format.
    """
    if os.path.exists("../../../data/NCTE/ncte.csv"):
        warnings.warn("Preprocessed NCTE dataset found. Loading from file.")
        return pd.read_csv("../../../data/NCTE/ncte.csv")
    if not os.path.exists("../../../data/NCTE/raw/ncte_single_utterances.csv") or not os.path.exists("../../../data/NCTE/raw/paired_annotations.csv") or not os.path.exists("../../../data/NCTE/raw/student_reasoning.csv"):
        raise FileNotFoundError("Please download the NCTE dataset and place the files in the ../../../data/NCTE/raw/ directory.")
    ncte = pd.read_csv("../../../data/NCTE/raw/ncte_single_utterances.csv")
    paired = pd.read_csv("../../../data/NCTE/raw/paired_annotations.csv")
    acts = []
    acts += paired.apply(lambda x: {"OBSID":int(x["exchange_idx"].split("_")[0]), "turn_idx":int(x["exchange_idx"].split("_")[1]), "acts": ["student on task" if x["student_on_task"]==1 else ""]},axis=1).tolist()
    acts += paired.apply(lambda x: {"OBSID":int(x["exchange_idx"].split("_")[0]), "turn_idx":int(x["exchange_idx"].split("_")[1])+1, "acts": ["teacher on task" if x["teacher_on_task"]==1 else "","high uptake" if x["teacher_on_task"]==1 else "","focusing question" if x["focusing_question"]==1 else ""]},axis=1).tolist()
    acts = pd.DataFrame(acts)
    acts["acts"] = acts["acts"].apply(lambda x: [y for y in x if y!=""])
    acts = acts[acts["acts"].apply(lambda x: len(x)>0)]
    reasoning = pd.read_csv("../../../data/NCTE/raw/student_reasoning.csv")
    reasoning = pd.DataFrame(reasoning.apply(lambda x: {"OBSID":int(x["comb_idx"].split("_")[0]), "turn_idx":int(x["comb_idx"].split("_")[1]),"reasoning": ["student reasoning" if x["student_reasoning"]==1 else ""]},axis=1).tolist())
    reasoning = reasoning[reasoning["reasoning"].apply(lambda x: x!=[""])]
    merged = acts.merge(reasoning, how="outer",on=["OBSID","turn_idx"]).fillna(value=-1)
    merged["final_acts"] = merged.apply(lambda x: (x["acts"] if x["acts"]!=-1 else [])+(x["reasoning"] if x["reasoning"]!=-1 else []),axis=1)
    merged["final_acts"] = merged["final_acts"].apply(lambda x: [y for y in x if y!=""])
    merged = merged.drop(columns=["acts","reasoning"]).rename(columns={"final_acts":"acts"})
    ncte  = ncte.merge(merged,how="left",on=["OBSID","turn_idx"]).fillna(-1)
    ncte["acts"] = ncte["acts"].apply(lambda x: x if x!=-1 else ["None"])
    df = []
    for id, grp in ncte.groupby("OBSID"):
        grp = grp.sort_values("turn_idx")
        df.append({**{"speakers":grp["speaker"].to_list(),"turns":grp["text"].tolist(),"dialogic_acts":grp["acts"].tolist(), "edu_level":"elementary"},** metadata["ncte"]})
    df = pd.DataFrame(df)
    if save:
        warnings.warn("Saving preprocessed NCTE dataset to ../../../data/NCTE/ncte.csv")
        df.to_csv("../../../data/NCTE/ncte.csv",index=False)
    else:
        warnings.warn("Preprocessed NCTE dataset not saved. To save it, set save=True.")
    return df

# %%
import re

def classify_grade(s: str):
    if not isinstance(s, str):
        print(s)
        return None
    s = s.strip()
    if s == "MS":
        return "middle"
    if s == "HS":
        return "high"
    if s=="4th & 5th":
        return "elementary"
    match = re.match(r"^(\d+)(st|nd|rd|th)$", s)
    if match:
        num = int(match.group(1))
        if num <= 5:
            return "elementary"
        elif 6 <= num <= 8:
            return "middle"
        elif num >= 9:
            return "high"
    return None


def process_file(path):
    d = pd.read_excel("../../../data/TalkMoves/raw/"+ path + ".xlsx").dropna(subset=["Sentence"])
    d["Teacher Tag"] = d["Teacher Tag"].apply(lambda x: "1 - none" if x==1 else "2 - keeping everyone together" if x==2 else x)
    d["Student Tag"] = d["Student Tag"].apply(lambda x: "1 - none" if x==1 else "5 - providing evidence/providing reasoning" if x==5  else " 2 - relating to another student" if x==2 else x)
    d = d.fillna("")
    try:
        return {"turns":d["Sentence"].tolist(),
        "speakers":d["Speaker"].tolist() if "Speaker" in d.columns else d["S"].tolist(),
        "dialogic_acts": d.apply(lambda x: x["Teacher Tag"].split("-")[1].strip()  if x["Teacher Tag"].strip()!="" else x["Student Tag"].split("-")[1].strip() if x["Student Tag"].strip()!="" else "none",axis=1).tolist()
        }
    except:
        warnings.warn("Error processing file: {}. Skipping.".format(path))
        return {"turns":None,
        "speakers":None,
        "dialogic_acts":None
        }

    

def match_file_name(name, files):
    distances = [distance(name,f) for f in files]
    min_distance = min(distances)
    if min_distance > 5:
        warnings.warn("No close match found for file name: {}. Skipping.".format(name))
        return None
    return files[distances.index(min_distance)]
    
def load_talkmoves(save=True):
    """
    Loads the TalkMoves dataset, processes the raw files, and returns a dataframe with the standardized format. If the preprocessed file already exists, it loads it directly. Otherwise, it processes the raw files.
    Args:
        save (bool): Whether to save the preprocessed dataframe as a CSV file for future use. Default is True.  
    Returns:
         pd.DataFrame: A dataframe containing the TalkMoves dataset with standardized format.  
    """
    if os.path.exists("../../../data/TalkMoves/talkmoves.csv"):
        return pd.read_csv("../../../data/TalkMoves/talkmoves.csv")
    if not os.listdir("../../../data/TalkMoves/raw/"):
        raise FileNotFoundError("Please download the TalkMoves dataset and place the excel files from subset 1 and 2 in the ../../../data/TalkMoves/raw directory.")
    if not os.path.exists("../../../data/TalkMoves/Datasheet.xlsx"):
        raise FileNotFoundError("Please save the Datasheet file from TalkMoces as ../../../data/TalkMoves/Datasheet.xlsx .")
    datasheet = pd.read_excel("../../../data/TalkMoves/Datasheet.xlsx",sheet_name=None)
    naming_dict = {"Name of File ":"file","Grade (exact grade or elementary, middle, high school)":"edu_level","Lesson Title":"file"}
    datasheet = pd.concat([list(datasheet.values())[0][["Name of File ","Grade (exact grade or elementary, middle, high school)"]].rename(columns=naming_dict),list(datasheet.values())[1][["Lesson Title","Grade (exact grade or elementary, middle, high school)"]].rename(columns=naming_dict)],ignore_index=True)
    datasheet["edu_level"] = datasheet.apply(lambda x: classify_grade(x["edu_level"]),axis=1)
    files = [x[:-5] for x in os.listdir("../../../data/TalkMoves/raw/")]
    datasheet["file"] = datasheet.apply(lambda x: match_file_name(x["file"], files),axis=1)
    datasheet = datasheet.dropna(subset=["file","edu_level"])
    df = datasheet.apply(lambda x: {**process_file(x["file"]), **metadata["talkmoves"], "edu_level":x["edu_level"]},axis=1).dropna().tolist()
    """drop files with None values in turns, speakers or dialogic_acts"""
    df = [x for x in df if x["turns"] is not None and x["speakers"] is not None and x["dialogic_acts"] is not None] 
    df = pd.DataFrame(df)
    if save:
        df.to_csv("../../../data/TalkMoves/talkmoves.csv",index=False)
    else:
        warnings.warn("Preprocessed TalkMoves dataset not saved. To save it, set save=True.")
    return df

# %%
def load_eedi(save=True):
    """Loads the Eedi dataset from HuggingFace, and returns a dataframe with the standardized format. If the preprocessed file already exists, it loads it directly. Otherwise, it processes the raw dataset.
    Args:
    save (bool): Whether to save the preprocessed dataframe as a CSV file for future use. Default is True.  
    Returns:
    pd.DataFrame: A dataframe containing the Eedi dataset with standardized format.  
    """
    if os.path.exists("../../../data/Eedi/eedi.csv"):
        return pd.read_csv("../../../data/Eedi/eedi.csv")
    ds = load_dataset("Eedi/Question-Anchored-Tutoring-Dialogues-2k", "anchored-dialogues")
    df = pd.concat([ds["train"].to_pandas(),ds["test"].to_pandas()],ignore_index=True)
    data = []
    for int_id,grp in df.groupby("InterventionId"):
        turns,speakers,acts = [],[],[]
        for idx, row in grp.iterrows():
            if row["IsTutor"]==1:
                speakers.append("tutor")
            else:
                speakers.append("student")
            turns.append(row["MessageString"])
            acts.append(row["TalkMovePrediction"][1:-1] if isinstance(row["TalkMovePrediction"], str) else "None")
        data.append({"turns":turns,"speakers":speakers,"dialogic_acts":acts, **metadata["eedi"], "edu_level":"middle"})
    df = pd.DataFrame(data)
    if save:
        df.to_csv("../../../data/Eedi/eedi.csv",index=False)
    else:
        warnings.warn("Preprocessed Eedi dataset not saved. To save it, set save=True.")
    return df

# %%

def process_mathdial_conversation(conv):
    turns, speakers, acts = [],[],[]
    for turn in conv.split("|EOM|"):
        speaker, utterance = turn.split(":",1)
        if len(utterance.strip())==0:
            continue
        speakers.append(speaker.strip())
        if speaker.strip().lower() == "teacher":
            turn = utterance.split(")",1)[1].strip()
            act = utterance.split(")",1)[0].strip(" (")
        else:
            turn = utterance.strip()
            act = "None"    
        turns.append(turn)
        acts.append(act)
    return {"turns":turns, "speakers":speakers,"dialogic_acts":acts}

def load_mathdial(save=True):
    """Loads the MathDial dataset from HuggingFace, and returns a dataframe with the standardized format. If the preprocessed file already exists, it loads it directly. Otherwise, it processes the raw dataset.
    Args:
    save (bool): Whether to save the preprocessed dataframe as a CSV file for future use. Default is True.  
    Returns:
    pd.DataFrame: A dataframe containing the MathDial dataset with standardized format.  
    """
    if os.path.exists("../../../data/MathDial/mathdial.csv"):
        return pd.read_csv("../../../data/MathDial/mathdial.csv")
    ds = load_dataset("eth-nlped/mathdial")
    df = pd.concat([ds["train"].to_pandas(),ds["test"].to_pandas()],ignore_index=True)
    data = df.apply(lambda x: {**process_mathdial_conversation(x["conversation"]),**metadata["mathdial"],"edu_level":"middle","extra":x["teacher_described_confusion"]},axis=1).tolist()
    df = pd.DataFrame(data)
    if save:   
        df.to_csv("../../../data/MathDial/mathdial.csv",index=False)
    else:
        warnings.warn("Preprocessed MathDial dataset not saved. To save it, set save=True.")
    return df

# %%
def load_tscc(save=True):
    """Loads the TSCC dataset from raw files, and returns a dataframe with the standardized format. If the preprocessed file already exists, it loads it directly. Otherwise, it processes the raw dataset.
    Args:
    save (bool): Whether to save the preprocessed dataframe as a CSV file for future use. Default is True.  
    Returns:
    pd.DataFrame: A dataframe containing the TSCC dataset with standardized format.  
    """
    if os.path.exists("../../../data/Tscc/tscc.csv"):
        return pd.read_csv("../../../data/Tscc/tscc.csv")
    if len(os.listdir("../../../data/Tscc/raw/"))==0:
        raise FileNotFoundError("Please download the TSCC dataset and place the files in the ../../../data/Tscc/raw/ directory.")
    files = os.listdir("../../../data/Tscc/raw/")
    data = []
    for file in files:
        d = pd.read_csv("../../../data/Tscc/raw/"+file,sep="\t")
        data.append(
            {"turns" : d["edited"].tolist(),
            "speakers" : d["role"].tolist(),
            "dialogic_acts" : d["seq.type"].fillna("None").apply(lambda x: x.split(",")).tolist(),
            **metadata["tscc"],
            "edu_level":None,
            "extra":d["focus"].tolist()})
    df = pd.DataFrame(data)
    if save:
        df.to_csv("../../../data/Tscc/tscc.csv",index=False)
    else:
        warnings.warn("Preprocessed TSCC dataset not saved. To save it, set save=True.")
    return df   

# %%

def load_weighttaks(save=True):
    """Loads the WeightTasks dataset from raw files, and returns a dataframe with the standardized format. If the preprocessed file already exists, it loads it directly. Otherwise, it processes the raw dataset.
    Args:
    save (bool): Whether to save the preprocessed dataframe as a CSV file for future use. Default is True.  
    Returns:
    pd.DataFrame: A dataframe containing the WeightTasks dataset with standardized format.  
    """
    if os.path.exists("../../../data/WeightTasks/weighttasks.csv"):
        return pd.read_csv("../../../data/WeightTasks/weighttasks.csv")
    if len(os.listdir("../../../data/WeightTasks/raw/"))==0:
        raise FileNotFoundError("Please download the WeightTasks dataset and place the files in the ../../../data/WeightTasks/raw/ directory.")
    files = [x for x in os.listdir("../../../data/WeightTasks/raw/") if x.startswith("Group")]
    annotations = pd.read_csv("../../../data/WeightTasks/raw/All_Groups_CPS.csv").fillna(0)
    annotations["Group"] = annotations["utteranceID"].apply(lambda x: int(x.split("_")[1]))
    annotations["Utterance"] = annotations["utteranceID"].apply(lambda x: int(x.split("_")[2]))
    taxonomy = [x for x in list(annotations.columns) if x.startswith("CPS")]
    annotations["acts"] = annotations.apply(lambda x : [taxonomy[i].replace("CPS_","") for i in range(len(taxonomy)) if x[taxonomy[i]]==1],axis=1)
    annotations["acts"] = annotations["acts"].apply(lambda x: x if len(x)>0 else ["None"])
    data = []
    for file in files:
        df = pd.read_csv("../../../data/WeightTasks/raw/"+file)
        df = df.merge(
            annotations[["Group", "Utterance", "acts"]],
            on=["Group", "Utterance"],
            how="left"
        )
        data.append({**{"speakers":df["Participant"].to_list(),"turns":df["Transcript"].tolist(),"dialogic_acts":df["acts"].tolist(), "edu_level":"university"},** metadata["weighttasks"]})
    df = pd.DataFrame(data)
    if save:
        df.to_csv("../../../data/WeightTasks/weighttasks.csv",index=False)
    else:
        warnings.warn("Preprocessed WeightTasks dataset not saved. To save it, set save=True.")
    return df
        
def load_delidata(save=True):
    ds = load_dataset("gkaradzhov/DeliData")
    df = ds["train"].to_pandas()
    df = df[df["message_type"]=="MESSAGE"]
    df["annotation_type"] = df["annotation_type"].fillna("None").apply(lambda x: x.replace("-"," ").replace("deliberation","").strip())
    df["annotation_target"] = df["annotation_target"].fillna("0").apply(lambda x: x.replace("0",""))
    df["acts"]  = df.apply(lambda x: x["annotation_type"] + (" - "+x["annotation_target"]) if x["annotation_target"]!="" else "",axis=1)
    df["acts"] = df["acts"].apply(lambda x: x if x!="" else "None")
    data = []
    for ind,grp in df.groupby("group_id"):
        data.append({
            "speakers": grp["origin"].tolist(),
            "turns":grp["original_text"].tolist(),
            "dialogic_acts":grp["acts"].tolist(),
            "edu_level":"university",
            **metadata["delidata"]
        })
    df = pd.DataFrame(data)
    if save:
        df.to_csv("../../../data/DeliData/delidata.csv",index=False)
    else:
        warnings.warn("Preprocessed DeliData dataset not saved. To save it, set save=True.")
    return df
    


# %%
"""
    Main function run from terminal to load and process datasets.
    It takes command line arguments:
    --dataset which can be one of "ncte","talkmoves","eedi","tscc","mathdial","weighttasks" or "all" to load all datasets and concatenate them. If not specified, it defaults to all.
    --save which can be True or False to specify whether to save the preprocessed datasets as CSV files for future use. If not specified, it defaults to True.
    --path which specifies the path to save the preprocessed datasets. If not specified, it defaults to ../../../data/processed_datasets.csv. If save is False, this argument is ignored.
    --reset which can be True or False to specify whether to reprocess the datasets even if the preprocessed files already exist. If not specified, it defaults to False, meaning that if the preprocessed files are found, they will be loaded directly without reprocessing.
"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and preprocess educational dialogue datasets.")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset to load: ncte, talkmoves, eedi, tscc, mathdial, weighttasks or all. Default is all.")
    parser.add_argument("--save", type=bool, default=True, help="Whether to save the preprocessed datasets as CSV files for future use. Default is True.")
    parser.add_argument("--path", type=str, default="../../../data/processed_datasets.csv", help="Path to save the preprocessed datasets if --save is True. Default is ../../../data/processed_datasets.csv.")
    parser.add_argument("--reset", type=bool, default=False, help="Whether to reprocess the datasets even if the preprocessed files already exist. Default is False.")
    args = parser.parse_args()
    
    if args.dataset not in DATASETS and args.dataset!="all":
        raise ValueError("Invalid dataset argument. Please choose from ncte, talkmoves, eedi, tscc, mathdial, weighttasks or all.")
    
    if args.reset:
        """
            Remove any csv file that is under a subdirectory of ../../../data/ (but not those in subsubdirectory raw) to force reprocessing of the datasets.
        """
        for root, dirs, files in os.walk("../../../data/"):
            if "raw" in root:
                continue
            for file in files:
                if file.endswith(".csv"):
                    os.remove(os.path.join(root, file))
                    print(f"Removed preprocessed file: {os.path.join(root, file)}")
    if args.dataset == "all":
        df_ncte = load_ncte(save=args.save)
        df_talkmoves = load_talkmoves(save=args.save)
        df_eedi = load_eedi(save=args.save)
        df_tscc = load_tscc(save=args.save)
        df_mathdial = load_mathdial(save=args.save)
        df_weighttasks = load_weighttaks(save=args.save)
        df_deli = load_delidata(save=args.save)
        final_df = pd.concat([df_ncte, df_talkmoves, df_eedi, df_tscc, df_mathdial, df_weighttasks,df_deli], ignore_index=True)
    else:
        if args.dataset == "ncte":
            final_df = load_ncte(save=args.save)
        elif args.dataset == "talkmoves":
            final_df = load_talkmoves(save=args.save)
        elif args.dataset == "eedi":
            final_df = load_eedi(save=args.save)
        elif args.dataset == "tscc":
            final_df = load_tscc(save=args.save)
        elif args.dataset == "mathdial":
            final_df = load_mathdial(save=args.save)
        elif args.dataset == "weighttasks":
            final_df = load_weighttaks(save=args.save)
        elif args.dataset == "delidata":
            final_df = load_delidata(save=args.save)

    if args.save:
        final_df.to_csv(args.path,index=False)
        print(f"Preprocessed dataset saved to {args.path}")
    else:
        warnings.warn("Preprocessed dataset not saved. To save it, set --save True when running the script.")


