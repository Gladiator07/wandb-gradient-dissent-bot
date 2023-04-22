import os
from dataclasses import asdict

import pandas as pd
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm
from wandb.integration.langchain import WandbTracer

import wandb
from config import config


def get_data(artifact_name: str = "gladiator/gradient_dissent_bot/summary_que_data:latest"):
    podcast_artifact = wandb.use_artifact(artifact_name, type="dataset")
    podcast_artifact_dir = podcast_artifact.download(config.root_data_dir)
    filename = artifact_name.split(":")[0].split("/")[-1]
    df = pd.read_csv(os.path.join(podcast_artifact_dir, f"{filename}.csv"))
    return df


def create_embeddings(episode_df: pd.DataFrame):
    # load docs into langchain format
    loader = DataFrameLoader(episode_df, page_content_column="transcript")
    data = loader.load()

    # split the documents
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    title = data[0].metadata["title"]
    print(f"Number of documents for podcast {title}: {len(docs)}")

    # initialize embedding engine
    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=os.path.join(config.chromadb_dir, title.replace(" ", "_")),
    )
    db.persist()


if __name__ == "__main__":
    # initialize wandb tracer
    WandbTracer.init(
        {
            "project": "gradient_dissent_bot",
            "name": "embed_transcripts",
            "job_type": "embed_transcripts",
            "config": asdict(config),
        }
    )

    # get data
    df = get_data(artifact_name=config.summarized_que_data_artifact)

    # create embeddings
    with get_openai_callback() as cb:
        for episode in tqdm(df.iterrows(), total=len(df), desc="Embedding transcripts"):
            episode_data = episode[1].to_frame().T

            create_embeddings(episode_data)

        print("*" * 25)
        print(cb)
        print("*" * 25)

        wandb.log(
            {
                "total_prompt_tokens": cb.prompt_tokens,
                "total_completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost,
            }
        )

    # log embeddings to wandb artifact
    artifact = wandb.Artifact("transcript_embeddings", type="dataset")
    artifact.add_dir(config.chromadb_dir)
    wandb.log_artifact(artifact)

    WandbTracer.finish()
