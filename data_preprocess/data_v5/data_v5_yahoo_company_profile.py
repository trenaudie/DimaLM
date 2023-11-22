# get the company profile name for every company ticker, using Selenium
from json import dump
from bs4 import BeautifulSoup
import requests
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import chain

ROOT_DIR = next(
    filter(
        lambda s: "LLM" in s.name, chain(Path().absolute().parents, [Path(os.getcwd())])
    ),
    None,
)
print(f"ROOT_DIR {ROOT_DIR}")
import configparser
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read(ROOT_DIR / "config.ini")
DATA_FOLDER = Path(config.get("paths", "path_data"))
print(f"DATA_FOLDER {DATA_FOLDER}")
assert os.path.isdir(DATA_FOLDER)
assert os.path.isdir(ROOT_DIR)
import sys

sys.path.append(str(ROOT_DIR))
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from utils.io_utils import write_headlines_to_file
from concurrent.futures import ThreadPoolExecutor, wait
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys


def get_yahoo_finance_description(ticker_symbol):
    # Build the URL of the target page
    url = f"https://finance.yahoo.com/quote/{ticker_symbol}/profile"

    # Initialize a web driver instance to control a Chrome window
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    # Set up the window size of the controlled browser
    driver.set_window_size(1920, 1080)

    # Visit the target page
    driver.get(url)

    try:
        # Wait until the description element is located and loaded
        # Use an XPath selector to target the description paragraph
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, "//p[contains(@class, 'Mt(15px)')]")
            )
        )
        description = element.text
    except Exception as e:
        description = f"Description not found due to error: {e}"
    # Close the browser and free up the resources
    driver.quit()

    return description


if __name__ == "__main__":
    dfnews = pd.read_parquet(ROOT_DIR / "temp/data_v5_ticker_done.parquet")
    company_profiles = {ticker: None for ticker in dfnews["ticker"].unique()}

    def get_result(ticker, future, i: int | None = None):
        """Helper function to process results."""
        if i is not None:
            print(f"i {i}, ticker {ticker}")
        company_profiles[ticker] = future.result()

    # List to keep track of futures
    futures = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        for i, ticker in enumerate(tqdm(dfnews["ticker"].unique())):
            future = executor.submit(get_yahoo_finance_description, ticker)
            # Add callback to process result when it's ready
            future.add_done_callback(lambda f, t=ticker: get_result(t, f, i))
            futures.append(future)

            if i % 100 == 0:
                # Waiting for all futures up to this point to finish
                wait(futures)
                with open(
                    ROOT_DIR / f"temp/company_profiles_yahoo_ckpt{i}.txt", "w"
                ) as f:
                    dump(company_profiles, f)
                # Resetting the futures list
                futures = []

    # Add the company profiles to the dataframe
    dfnews["company_profile"] = dfnews["ticker"].apply(lambda x: company_profiles[x])
    dfnews.to_parquet(ROOT_DIR / "temp" / "data_v5_yahoo_company_profile_done.parquet")
    print(f"company profiles ")
    for i, company_profile in enumerate(dfnews["company_profile"].unique()):
        print(f"{i} {company_profile}")
    write_headlines_to_file(dfnews=dfnews, add_cols=["company_profile"])
