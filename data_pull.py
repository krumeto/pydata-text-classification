# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets==4.0.0",
#     "marimo>=0.14.10",
#     "pandas==2.3.1",
#     "scikit-learn==1.7.1",
# ]
# ///

import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Due to `datasets` being a mess of a package, run this notebook with
    `uvx marimo edit --sandbox data_pull.py`
    If you are interested, check `uvx` and the shebang at the start of the notebook.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## This notebook prepares the training, validation and test datasets for the talk.

    The dataset that we will use is [Prototypical HAI collaborations](https://huggingface.co/datasets/microsoft/prototypical-hai-collaborations) by Microsoft, from the 2025 paper: [Prototypical Human-AI Collaboration Behaviors from LLM-Assisted Writing in the Wild](https://arxiv.org/abs/2505.16023)

    We are going to use the following subset:

    - `wildchat1m_en3u-task_anns.jsonl`: The sessions in WildChat-1M which are in English and contain 3 utterances. This contains 159134 user-LLM conversation sessions. Each session is annotated with one or more "coarse tasks" from GPT-4o. Appendix B.1 and B.2 from the paper contains details of how the labels are predicted.

    In addition, we are keeping the top 5 types of conversations, in order to keep the focus on the text classification frameworks
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from datasets import load_dataset

    # see here for a dataset card https://huggingface.co/datasets/microsoft/prototypical-hai-collaborations
    DATASET_NAME = "microsoft/prototypical-hai-collaborations"
    SUBSET = "wildchat1m_en3u-task_anns"

    SAMPLE_LABELS = [
        ["GENERATING COMMUNICATIVE TEXT"],
        ["INFORMATION SEARCH"],
        ["SOFTWARE DEVELOPMENT"],
        ["GENERATING CREATIVE TEXT"],
        ["HOMEWORK PROBLEM"],
    ]

    ENGLISH_SPEAKING_COUNTRIES = [
        "United States",
        "United Kingdom",
        "Canada",
        "New Zealand",
        "Australia",
    ]

    VAL_SET_SIZE = 0.15
    TEST_SET_SIZE = 0.15
    return (
        DATASET_NAME,
        ENGLISH_SPEAKING_COUNTRIES,
        SAMPLE_LABELS,
        SUBSET,
        TEST_SET_SIZE,
        VAL_SET_SIZE,
        load_dataset,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Pull and preprocessing

    We pull a portion of the dataset, fetch only conversations labelled with the desired categories and from the respective countries.
    """
    )
    return


@app.cell
def _(
    DATASET_NAME,
    ENGLISH_SPEAKING_COUNTRIES,
    SAMPLE_LABELS,
    SUBSET,
    load_dataset,
):
    # helper functions
    def flatten_conversation(batch):
        """
        Convert a list of {'author': ..., 'utterance': ...} into a single string
        with 'USER: ' and 'ASSISTANT: ' prefixes.
        """
        flattened = []
        for convo in batch["turns"]:
            parts = []
            for msg in convo:
                author = msg.get("author", "").lower()
                prefix = "USER: " if author == "user" else "ASSISTANT: "
                parts.append(f"{prefix}{msg.get('utterance', '').strip()}")
            flattened.append("\n".join(parts))
        return {"text": flattened}

    def flatten_list(example):
        example["coarse_tasks"] = example["coarse_tasks"][0]
        return example

    ds = (
        load_dataset(
            DATASET_NAME, SUBSET, split="train[:7500]"
        )  # starting with a bit more, as we are doing a lot of filtering below
        .filter(
            lambda row: row["coarse_tasks"] in SAMPLE_LABELS
        )  # fetch only the rows containing the sample labels
        .filter(lambda row: row["country"] in ENGLISH_SPEAKING_COUNTRIES)
        .map(
            flatten_conversation, batched=True
        )  # get a string representation of the conversations
        .map(flatten_list)  # get a string label, rather than a list[str]
        .select_columns(["text", "coarse_tasks"])
        .rename_column("coarse_tasks", "label")
    )

    ds
    return (ds,)


@app.cell
def _(ds):
    ds.to_pandas()
    return


@app.cell
def _(ds):
    ds.to_pandas().label.value_counts().round(2)
    return


@app.cell
def _(mo):
    mo.md(r"""## Data split""")
    return


@app.cell
def _(TEST_SET_SIZE, VAL_SET_SIZE, ds):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from typing import Tuple

    def get_splits(
        df: pd.DataFrame,
        test_size: float,
        eval_size: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Do a two-step split: first into train+eval and test, then into train and eval
        train_and_eval_size = 1.0 - test_size
        train_and_eval_df, test_df = train_test_split(
            df,
            test_size=test_size,
            train_size=train_and_eval_size,
            stratify=df["label"],
            random_state=42,
        )
        relative_eval_size = eval_size / train_and_eval_size
        train_size = 1.0 - relative_eval_size
        train_df, eval_df = train_test_split(
            train_and_eval_df,
            test_size=relative_eval_size,
            train_size=train_size,
            stratify=train_and_eval_df["label"],
            random_state=42,
        )

        return (train_df, test_df, eval_df)

    train_df, test_df, eval_df = get_splits(
        df=ds.to_pandas(), eval_size=VAL_SET_SIZE, test_size=TEST_SET_SIZE
    )
    return eval_df, test_df, train_df


@app.cell
def _(eval_df, test_df, train_df):
    import os

    os.makedirs("data", exist_ok=True)

    # A bit more convenient in a dict with names
    dataframes = {"train": train_df, "test": test_df, "eval": eval_df}

    for name, df in dataframes.items():
        path = os.path.join("data", f"{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"Saved → {path!r}")

        print(f"\n{name.capitalize()} DataFrame:")
        print(f"  • Rows:    {df.shape[0]}")
        print(f"  • Columns: {df.shape[1]}")
        print("\nInfo:")
        df.info()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
