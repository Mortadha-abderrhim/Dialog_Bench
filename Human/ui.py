import streamlit as st
import pandas as pd
import glob
import ast
import jsonlines
st.set_page_config(layout="wide")



if "counter" not in st.session_state:
    ## get all files with .csv in some directory
    files = glob.glob("data/*/*.csv")
    nb_examples = 1
    st.session_state.counter = -1
    st.session_state.data = [
        pd.read_csv(file).sample(n=nb_examples) for file in files
    ]
    st.session_state.prompts = [
        {
            "learning_context":"Based on the given segment of dilaogue, determine the instructional/social learning context represented in the dialogue.\n\nOPTIONS\n\nA) Classroom Instruction\na teacher addressing a whole class or leading a lesson.\n\nB) Tutoring\none-on-one guidance between a tutor/teacher and a student.\n\nC) Collaborative Group Activity\nmultiple participants (e.g., students) working together as peers\n\n",
            "comm_modality":"Based on the given segment of dialogue, determine the communication modality represented in the interaction.\n\nOPTIONS\n\nA) In-Person\nparticipants are physically co-present.\n\nB) Synchronous Online Spoken\nreal-time spoken interaction via audio/video (e.g., Zoom call).\n\nC) Synchronous Text-Based Chat\nreal-time written exchange (e.g., live chat, messaging).\n\nD) Asynchronous Online Communication\ndelayed written interaction (e.g., forum posts, emails).\n\n",
            "agent_config": "Based on the given segment of dialogue, determine the agent configuration represented in the interaction.\n\nOPTIONS\n\nA) Human-Human\nall participants are human (e.g., student, teacher, peers).\n\nB) Human-AI\nat least one participant is an AI system (e.g., chatbot, virtual tutor).\n\n",
            "subject": "Based on the given segment of dialogue, determine the subject domain of the learning interaction.\n\nReturn the most appropriate subject domain as a short lowercase label, examples: mathematics, language learning...",
            "edu_level":"Based on the given segment of dialogue, determine the educational level associated with the learning interaction.\n\nOPTIONS\n\nA) Primary\nearly/basic education (typically elementary school).\n\nB) Middle\nmiddle school or early secondary education.\n\nC) High\nadvanced secondary education (e.g., high school).\n\nD) University\nhigher education (e.g., college, undergraduate, graduate).\n\n"
        },
        {
            "speaker_id":"Based on the given dialogue chunk and the next turn, determine which existing speaker most likely said the next turn. Write only the ID.\nNext turn: \n\t{NEXT_TURN}",
            "teacher_id": "Based on the given dialogue, determine which speaker is acting as the teacher or tutor. Write only the ID.",
            "next_turn": "Based on the given dialogue chunk, determine which option is the most likely next turn in the conversation.",
            "previous_turn":"Based on the given dialogue chunk, determine which option is the most likely previous turn in the conversation.",
            "chunk_ordering": "Based on the two dialogue chunks, determine their chronological order in the conversation.\n\nOPTIONS\n\nA) Chunk A happened before Chunk B.\n\nB) Chunk B happened before Chunk A."
        },
        {
            "acts_annotation": {
                "single": "Based on the dialogic act taxonomy and the annotated dialogue chunk, determine the most appropriate dialogic act for the next turn.\nNext turn: \n\t{NEXT_TURN}",
                "multiple": "Based on the dialogic act taxonomy and the annotated dialogue chunk, determine the most appropriate dialogic act label(s) for the next turn.\n ** Multiple labels may apply! **.\nNext turn: \n\t{NEXT_TURN}"
            },
            "act_prediction":{
                "single": "Based on the dialogic act taxonomy and the annotated dialogue chunk, predict the most likely dialogic act of the next turn.",
                "multiple": "Based on the dialogic act taxonomy and the annotated dialogue chunk, determine the most appropriate dialogic act label(s) for the next turn.\b ** Multiple labels may apply! **."

            }
        }
    ]

def next_callback(example):
    with jsonlines.open("answers/answers.jsonl", mode='a') as writer:
        writer.write(example.to_dict())
    st.session_state.update(counter=st.session_state.counter+1)


if st.session_state.counter >= len(st.session_state.data):
    st.title("All done!")
    st.text("Thank you for your participation.")
    st.stop()
else:
    example = st.session_state.data[st.session_state.counter].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        if example["column"] == "chunk_ordering":
            st.title("Dialogue Chunks")
            chunks = ast.literal_eval(example["chunks"])
            st.text("Chunk A:\n" + chunks[0] + "\n\nChunk B:\n" + chunks[1])
        elif example["column"] == "speaker_id":
            st.title("Dialogue")
            st.text(example["past"])
        else:
            st.title("Dialogue")
            st.text(example["dialogue"])
    with col2:
        boxes = None
        labels = None
        answer = None
        st.title("Task")
        if (example["dim"] == 1 and example["column"]!="subject") or example["column"]=="chunk_ordering":
            task = st.session_state.prompts[int(example["dim"])-1][example["column"]].split("\n\nOPTIONS\n\n")[0]
            options = st.session_state.prompts[int(example["dim"])-1][example["column"]].split("\n\nOPTIONS\n\n")[1].strip().split("\n\n")
            st.text(task)
            answer = st.radio(
                "Options:",
                options,
                index=None,
            )
            if answer is not None:
                answer = answer.strip()[0]
            
        elif example["column"]=="subject":
            task = st.session_state.prompts[int(example["dim"])-1][example["column"]]
            st.text(task)
            answer = st.text_input("Subject:")
        elif example["column"] == "speaker_id":
            task = st.session_state.prompts[1][example["column"]].format(NEXT_TURN=example["next"])
            st.text(task)
            answer = st.text_input("Speaker ID:")
        elif example["column"] == "teacher_id":
            task = st.session_state.prompts[1][example["column"]]
            st.text(task)
            answer = st.text_input("Teacher ID:")
        elif example["column"] in ["next_turn","previous_turn"]:
            task = st.session_state.prompts[int(example["dim"])-1][example["column"]]
            st.text(task)
            answer = st.radio(
                "Options:",
                ["ABCD"[ind]+")"+ option for ind,option in enumerate(ast.literal_eval(example["options"]))],
                index=None,
            )
            if answer is not None:
                answer = answer.strip()[0]
        elif example["column"] in ["acts_annotation", "act_prediction"]:
            number = "multiple" if example["dataset"] in ["ncte","tscc","weighttasks"] else "single"
            if example["column"] == "acts_annotation": 
                task = st.session_state.prompts[int(example["dim"])-1][example["column"]][number].format(NEXT_TURN=example["turn"])
            else:
                task = st.session_state.prompts[int(example["dim"])-1][example["column"]][number]
            st.text(task)
            if number == "multiple":
                st.text("Dialogic Act Taxonomy:\n")
                st.text(example["taxonomy"])
                labels = [act.split(":",1)[0] for act in example["taxonomy"].split("\n")]
                boxes = []
                options = st.multiselect(
                    "",
                    labels,
                    default=[],
                )
                answer = ", ".join(options)
            else:
                st.text("Dialogic Act Taxonomy:\n")
                acts = example["taxonomy"].split("\n")
                answer = st.radio(
                "",
                acts,
                index=None,
                )
                if answer is not None:
                    answer = answer.split(":",1)[0].strip()

        
        if answer is not None:
            example["generated"] = answer.strip()
        st.button("Next", on_click=next_callback,args = [example], disabled= (answer is None))
