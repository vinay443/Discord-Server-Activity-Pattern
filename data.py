import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from IPython.display import display
from typing import Any


# LOAD
def get_data() -> pd.DataFrame:
    """Loads and combines multiple CSV files into a single DataFrame.

    The function reads six CSV files from a predefined directory, concatenates
    them, and returns a combined DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame containing data from all six CSV files.
    """
    return pd.concat(
        [
            pd.read_csv(
                f"../data/VALORANT_General_679877028991860876_{i+1}.csv"
            )
            for i in range(6)
        ],
        ignore_index=True,
        sort=False,
    )


# CLEAN
def clean_column_names(df: pd.DataFrame):
    """Cleans and standardizes column names in a DataFrame.

    Converts column names to lowercase, removes extra spaces, and replaces spaces
    with underscores for better consistency.

    Args:
        df (pd.DataFrame): Input DataFrame whose column names need cleaning.
    """
    df.columns = [
        column.lower().strip().replace(" ", "_") for column in df.columns
    ]


def to_proper_types(df: pd.DataFrame):
    """Converts DataFrame columns to appropriate data types.

    This function sets:
        - 'authorid' as integer.
        - 'author', 'content', 'attachments', 'reactions' as strings.
        - 'date' as a datetime object.
    It also sorts the DataFrame by the 'date' column.

    Args:
        df (pd.DataFrame): Input DataFrame with raw data.
    """
    df["authorid"] = df["authorid"].astype(int)
    df["author"] = df["author"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["content"] = df["content"].astype(str)
    df["attachments"] = df["attachments"].astype(str)
    df["reactions"] = df["reactions"].astype(str)
    df.sort_values(by="date", inplace=True)


def drop_column(df: pd.DataFrame, col: str):
    """Drops a specified column from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Name of the column to drop.
    """
    df.drop(col, axis=1, inplace=True)


def drop_rows(df: pd.DataFrame, mask: pd.Series):
    """Drops rows from the DataFrame based on a boolean mask.

    Args:
        df (pd.DataFrame): Input DataFrame.
        mask (pd.Series): Boolean mask indicating rows to drop.
    """
    df.drop(df[mask].index, inplace=True)


def fill_na(df: pd.DataFrame, value: Any):
    """Fills missing values in the DataFrame with a specified value.

    Args:
        df (pd.DataFrame): Input DataFrame.
        value (Any): Value to use for filling missing entries.
    """
    df.fillna(value, inplace=True)


# EDA
def eda(df: pd.DataFrame):
    """Performs Exploratory Data Analysis (EDA) on the DataFrame.

    Outputs:
        - Shape of the DataFrame.
        - Sample rows from the DataFrame.
        - Column data types.
        - Missing values count.
        - Duplicate rows count.

    Args:
        df (pd.DataFrame): Input DataFrame to analyze.
    """
    print("1. How much data do I have?")
    print(df.shape)
    print()

    print("2. What does my data look like?")
    display(df.head())
    display(df.sample(5))
    print()

    print("3. What are the data types of each column?")
    display(df.info())
    print()

    print("4. Are there any missing values?")
    display(df.isnull().sum())
    print()

    print("5. Are there duplicate values?")
    display(df.duplicated().sum())
    print()


# Analysis
def add_sentiment(df: pd.DataFrame, textCol: str):
    """Adds a sentiment score column based on text analysis.

    The function uses VADER SentimentIntensityAnalyzer to compute a compound
    sentiment score for each message in the specified text column.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text data.
        textCol (str): Name of the column containing text for sentiment analysis.

    Returns:
        None: Adds a new 'sentiment' column to the DataFrame.
    """
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df[textCol].map(
        lambda txt: analyzer.polarity_scores(str(txt))["compound"]
    )


def add_date_derivatives(df: pd.DataFrame, dateCol: str):
    """Adds date-based derivative columns to the DataFrame.

    Derivative columns include:
        - Hour of the day.
        - Day of the week (0 = Monday).
        - Month of the year.
        - Year.
        - Day name (e.g., Monday).
        - Month name (e.g., January).

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime column.
        dateCol (str): Name of the datetime column for deriving new features.
    """
    df["hour"] = df[dateCol].dt.hour
    df["day"] = df[dateCol].dt.day_of_week
    df["month"] = df[dateCol].dt.month
    df["year"] = df[dateCol].dt.year
    df["day_name"] = df[dateCol].dt.day_name()
    df["month_name"] = df[dateCol].dt.month_name()


def add_total_reactions(df: pd.DataFrame, reactionsCol: str):
    """Calculates the total number of reactions for each row in a DataFrame.

    Extracts numeric values from a column containing reactions in string format
    and sums them to calculate total reactions.

    Args:
        df (pd.DataFrame): Input DataFrame containing the reactions column.
        reactionsCol (str): Name of the column storing reactions as strings.

    Returns:
        None: Adds a new 'reactions_total' column to the DataFrame.
    """
    df["reactions_total"] = (
        df[reactionsCol]
        .str.findall(r"\d+")
        .apply(lambda x: [int(i) for i in x])
        .apply(sum)
    )
