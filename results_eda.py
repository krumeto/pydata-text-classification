import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import shutil
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    import marimo as mo
    import pandas as pd


    return mo, os, pd, shutil


@app.cell
def _():
    TFIDF_METRICS = "models/tfidf/metrics_20250719_145737.json"
    MODEL2VEC_METRICS = "models/model2vec/metrics_20250719_145825.json"
    SETFIT_METRICS = "models/setfit/metrics_20250719_154815.json"
    return MODEL2VEC_METRICS, SETFIT_METRICS, TFIDF_METRICS


@app.cell
def _(MODEL2VEC_METRICS, SETFIT_METRICS, TFIDF_METRICS, pd):
    metrics_df = pd.DataFrame(
        {
            "tfidf": pd.read_json(TFIDF_METRICS, orient="index", typ="series"),
            "model2vec": pd.read_json(MODEL2VEC_METRICS, orient="index", typ="series"),
            "setfit": pd.read_json(SETFIT_METRICS, orient="index", typ="series")
        }
    )

    def highlight_per_row(s):
        # decide whether to look for min or max
        is_time_row = s.name in ['train_time_seconds', 'test_time_seconds']
        target = s.min() if is_time_row else s.max()
        # if there’s more than one occurrence of the target, don’t colour anything
        if (s == target).sum() == len(s):
            return ['' for _ in s]
    
        # otherwise, build a mask for the single target cell
        mask = s == target
        color = 'lightgreen'
        return [f'background-color: {color}' if v else '' for v in mask]\
    
    metrics_df.style.apply(highlight_per_row, axis=1)
    return


@app.cell
def _(mo):
    mo.md(r"""## Load the models from Huggingface""")
    return


@app.cell
def _(os, shutil):
    import joblib
    from skops.hub_utils import download

    ## The hard one - TFIDF

    TF_IDF_DST = 'models/tfidf/model/'
    # If it exists, delete it (and all its contents) because skops is stupid
    if os.path.isdir(TF_IDF_DST):
        shutil.rmtree(TF_IDF_DST)
    os.makedirs(TF_IDF_DST, exist_ok=True)

    download(repo_id="krumeto/text-class-tutorial-tfidf", dst=TF_IDF_DST)
    model_tfidf = joblib.load(
    	TF_IDF_DST + "/skops-dkftfzgw.pkl"
    )

    ## Model2vec
    from model2vec.inference import StaticModelPipeline

    model_model2vec = StaticModelPipeline.from_pretrained("krumeto/text-class-tutorial-model2vec")

    ## setfit
    from setfit import SetFitModel

    model_setfit = SetFitModel.from_pretrained("krumeto/text-class-tutorial-setfit")

    models_dict = {
        'tfidf': model_tfidf,
        'model2vec': model_model2vec,
        'setfit': model_setfit
    }
    return model_model2vec, model_setfit, model_tfidf, models_dict


@app.cell
def _(models_dict):
    test_string = "How do I define a string in Python?"


    for name, mod in models_dict.items():
        print(f"{name} scoring:")
        print(mod.predict([test_string]))
        print(mod.predict_proba([test_string]))

    return


@app.cell
def _(mo):
    mo.md(r"""## Load the test dataset and lets see what's going on""")
    return


@app.cell
def _(model_model2vec, model_setfit, model_tfidf, pd):
    test_data = (pd.read_parquet("data/test.parquet")
                .assign(
                    tfidf = lambda d: model_tfidf.predict(d.text),
                    model2vec = lambda d: model_model2vec.predict(d.text),
                    setfit = lambda d: model_setfit.predict(d.text, as_numpy=True)
                ))
    return (test_data,)


@app.cell
def _(test_data):
    def filter_agreements(dataf):
        other_cols = [c for c in dataf.columns if c != 'text']
        uniform_mask = dataf[other_cols].nunique(axis=1) > 1
        return dataf.loc[uniform_mask]

    filter_agreements(test_data)
    return


@app.cell
def _(model_setfit, test_data):
    model_setfit.predict_proba(test_data.text.iloc[155])
    return


@app.cell
def _(model_setfit):
    model_setfit.model_head.classes_
    return


@app.cell
def _(model_model2vec, test_data):
    model_model2vec.predict_proba(test_data.text.iloc[155]).round(3)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
