import os
import sys
import tokenizers
import transformers
import textwrap
import pandas as pd
import logging
import configparser
from pathlib import Path
from itertools import chain

ROOT_DIR = next(
    filter(
        lambda s: "LLM" in s.name, chain(Path().absolute().parents, [Path(os.getcwd())])
    ),
    None,
)
sys.path.append(ROOT_DIR)

from transformers import PreTrainedModel
from peft import PeftModel, PeftConfig
from transformers import PreTrainedModel


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,  # Set the logging level to INFO to see the trainer's logs
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

# If you wish to further configure the logger used in your script:
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  #


def write_log_to_file(data: list[dict[str, float]], filepath: str | os.PathLike):
    with open(filepath, "a") as fp:
        # Write header
        fp.write("epoch - step - tr_loss - eval_loss - eval_accuracy\n")

        # Iterate over data and write formatted content
        for i in range(
            0, len(data) - 1
        ):  # Assuming training and evaluation data alternate
            row = data[i]

            epoch = row.get("epoch", None)
            step = row.get("step", None)
            tr_loss = row.get("loss", None)
            eval_loss = row.get("eval_loss", None)
            eval_accuracy = row.get("eval_accuracy", None)
            metrics = {
                "tr_loss": tr_loss,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
            }

            for key, value in metrics.items():
                metrics[key] = f"{value:.4f}" if value is not None else "     "

            tr_loss = metrics["tr_loss"]
            eval_loss = metrics["eval_loss"]
            eval_accuracy = metrics["eval_accuracy"]

            fp.write(
                f"{epoch:.2f} - {step} - {tr_loss} - {eval_loss} - {eval_accuracy}\n"
            )


def load_peft_model_from_files(pretrained_model: PreTrainedModel, output_dir: str):
    checkpoint_name = next(
        filter(
            lambda x: x.startswith("checkpoint")
            and os.path.isdir(os.path.join(output_dir, x)),
            os.listdir(output_dir),
        )
    )
    assert checkpoint_name is not None, "No checkpoint found in output_dir"
    pretrained_model_name_or_path = os.path.join(output_dir, checkpoint_name)
    assert os.path.isdir(
        pretrained_model_name_or_path
    ), "No adapter_config.json found in output_dir"
    logger.info(f"loading peft adapter from {pretrained_model_name_or_path}")
    peftconfig = PeftConfig.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path
    )
    model = PeftModel.from_pretrained(
        model=pretrained_model,
        model_id=pretrained_model_name_or_path,
        config=peftconfig,
    )
    return model


def write_headlines_to_file(
    dfnews: pd.DataFrame,
    x_col: str = "headline",
    add_cols: list[str] = [],
    filepath: os.PathLike = "headline.txt",
    limit: int = 1000,
    step: int = 1,
    keep_date: bool = True,
):
    # function to log some headlines to file, to better analyze a given StreetAccount or Down Jones dataset

    MAX_LINE_LENGTH = 100
    data_col = next(filter(lambda x: "date" in x.lower(), dfnews.reset_index().columns))
    headline_col = next(filter(lambda x: x_col in x.lower(), dfnews.columns))
    from itertools import islice

    with open(filepath, "w") as fp:
        for row in islice(dfnews.reset_index().iterrows(), 0, limit, step):
            date = row[1][data_col]
            headline = row[1][headline_col]
            wrapped_headline = textwrap.fill(headline, MAX_LINE_LENGTH)
            for col in add_cols:
                wrapped_headline = f"{row[1][col]}--{wrapped_headline}"
            if keep_date:
                fp.write(f"{date} - {wrapped_headline}\n")
            else:
                fp.write(f"{wrapped_headline}\n")


class SuppressLogging:
    def __enter__(self):
        logging.disable(logging.CRITICAL + 1)

    def __exit__(self, exc_type, exc_value, traceback):
        logging.disable(logging.NOTSET)
