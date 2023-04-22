from dataclasses import dataclass


@dataclass
class Config:
    playlist_url: str = "https://www.youtube.com/playlist?list=PLD80i8An1OEEb1jP0sjEyiLG8ULRXFob_"

    # paths
    root_data_dir: str = "../data"
    yt_scraped_data_path: str = "../data/yt_data.csv"
    chromadb_dir: str = "../data/chromadb"

    # artifacts
    yt_podcast_data_artifact: str = "gladiator/gradient_dissent_bot/yt_podcast_data:latest"
    summarized_data_artifact: str = "gladiator/gradient_dissent_bot/summary_data:latest"
    summarized_que_data_artifact: str = "gladiator/gradient_dissent_bot/summary_que_data:latest"


config = Config()
