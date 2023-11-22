import pandas as pd


def gen_news(dfnews, cols_add: list[str] = [], cols_remove: list[str] = []):
    cols = ["DATE", "headline"] + cols_add
    cols = [col for col in cols if col not in cols_remove]
    for i in range(0, dfnews.shape[0], 10):
        yield dfnews.reset_index().iloc[i : i + 10].loc[:, cols]
