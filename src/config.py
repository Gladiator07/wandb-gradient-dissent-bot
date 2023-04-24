from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    playlist_url: str = "https://www.youtube.com/playlist?list=PLD80i8An1OEEb1jP0sjEyiLG8ULRXFob_"

    # paths
    root_data_dir: Path = Path("data")
    root_artifact_dir: Path = Path("downloaded_artifacts")

    # wandb
    project_name: str = "gradient_dissent_qabot"
    yt_podcast_data_artifact: str = "gladiator/gradient_dissent_qabot/yt_podcast_transcript:latest"
    summarized_data_artifact: str = "gladiator/gradient_dissent_qabot/summarized_podcasts:latest"
    summarized_que_data_artifact: str = (
        "gladiator/gradient_dissent_qabot/summarized_que_podcasts:latest"
    )


config = Config()
