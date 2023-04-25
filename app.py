import os
import re
from ast import literal_eval

import wandb
import gradio as gr
import pandas as pd
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from src.config import config

# download and read data
api = wandb.Api()
artifact_df = api.artifact(config.summarized_que_data_artifact)
artifact_df.download(config.root_data_dir)

artifact_embeddings = api.artifact(config.transcript_embeddings_artifact)
chromadb_dir = artifact_embeddings.download(config.root_data_dir / "chromadb")

df_path = config.root_data_dir / "summarized_que_podcasts.csv"
df = pd.read_csv(df_path)


def embed_video(title: str):
    video_url = df[df["title"] == title]["url"].values[0]
    match = re.search(r"v=([-\w]+)", video_url)
    video_id = match.group(1)
    # embed video
    # video_embed = f"<iframe width='600' height='330' src=https://www.youtube.com/embed/{video_id} frameborder='0' allowfullscreen></iframe>"
    video_embed = f"<iframe width='580' height='360' src=https://www.youtube.com/embed/{video_id} frameborder='0' allowfullscreen style='width:100%; max-width:100%;'></iframe>"

    return video_embed


def get_podcast_info(title: str):
    # get questions
    questions = df[df["title"] == title]["questions"].values[0]
    questions = literal_eval(questions)
    que_str = ""
    for que in questions:
        que_str += f"👉 {que}\n"

    # get summary
    summary = df[df["title"] == title]["summary"].values[0]

    return summary, que_str


def get_answer(podcast: str, question: str):
    index = df[df["title"] == podcast].index[0]
    db_dir = os.path.join(chromadb_dir, str(index))
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    prompt_template = """Use the following pieces of context to answer the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Don't add your opinions or interpretations. Ensure that you complete the answer.
    If the question is not relevant to the context, just say that it is not relevant.

    CONTEXT:
    {context}

    QUESTION: {question}

    ANSWER:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retriever = db.as_retriever()
    retriever.search_kwargs["k"] = 2

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    with get_openai_callback() as cb:
        result = qa({"query": question})
        print(cb)

    answer = result["result"]
    return answer


with gr.Blocks() as demo:
    # gr.Markdown("# Welcome to Gradient Dissent QA Bot!")
    gr.Markdown("<h1><center>Welcome to Gradient Dissent QA Bot 🤖</center></h1>")
    gr.Markdown(
        "#### The purpose of this QA bot is to provide answers to questions related to podcast episodes from Weights & Biases' [Gradient Dissent Podcast](https://www.youtube.com/playlist?list=PLD80i8An1OEEb1jP0sjEyiLG8ULRXFob_)."
    )
    gr.Markdown(
        "#### First select a podcast episode and click `Get Podcast Info` to get the summary and possible questions about the podcast episode."
    )
    gr.Markdown(
        "#### Then ask a question about the podcast episode and click `Get Answer` to get the answer."
    )
    gr.Markdown("<br>")

    with gr.Row():
        with gr.Column(scale=0.5):
            dropdown = gr.Dropdown(
                df["title"].to_list(), label="Select a Podcast Episode", value=df.iloc[0]["title"]
            )
            podcast_info_btn = gr.Button("Get Podcast Info")

            podcast_info_btn.click(
                fn=embed_video,
                inputs=dropdown,
                outputs=gr.HTML(label="Podcast Video"),
            )

            question_box = gr.Textbox(label="Ask a question about the podcast episode")
            with gr.Row():
                ques_clear_btn = gr.Button("Clear")
                ques_btn = gr.Button("Get Answer")

            ques_btn.click(
                fn=get_answer,
                inputs=[dropdown, question_box],
                outputs=gr.Textbox(label="Answer"),
            )
            ques_clear_btn.click(lambda: None, None, question_box, queue=False)

        with gr.Column(scale=0.5):
            podcast_info_btn.click(
                fn=get_podcast_info,
                inputs=dropdown,
                outputs=[
                    gr.Text(label="Summary of the podcast"),
                    gr.Text(label="Some of the questions you can ask"),
                ],
            )


demo.launch()
