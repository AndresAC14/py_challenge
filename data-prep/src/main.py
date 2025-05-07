import polars as pl
import httpx
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle as pk
from datetime import datetime


def get_data(data_points: int):
    """
    Get the schema and the full data and save them into two separate JSON files.
    """

    schema_response = httpx.get("http://127.0.0.1:8777/api/v1/animals/schema")
    data_response = httpx.post(
        "http://127.0.0.1:8777/api/v1/animals/data",
        json={"seed": 42, "number_of_datapoints": data_points},
    )

    # Check for request success
    if schema_response.status_code == 200 and data_response.status_code == 200:
        with open("data/schema.json", "w") as schema_out:
            json.dump(schema_response.json(), schema_out, indent=2)

        with open("data/animals.json", "w") as data_out:
            json.dump(data_response.json(), data_out, indent=2)
    else:
        raise Exception("Failed to fetch data from the API")


def create_dataframe() -> pl.DataFrame:
    """
    Transforms the JSON data into a Pandas DataFrame.
    """
    with open("data/animals.json", "r") as data_file:
        data = json.load(data_file)

    df = pl.DataFrame(data)

    return df


def remove_outliers_iqr(df: pl.DataFrame, column: str) -> pl.DataFrame:
    q1 = df.select(pl.col(column).quantile(0.25, "nearest")).item()
    q3 = df.select(pl.col(column).quantile(0.75, "nearest")).item()

    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return df.filter((pl.col(column) >= lower) & (pl.col(column) <= upper))


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Dummify boolean columns and remove outliers
    Remove outliers
    """

    df = df.with_columns(
        [
            pl.col(col).cast(pl.Float64)
            for col in df.columns
            if df.schema[col] == pl.Boolean
        ]
    )

    for col in ["walks_on_n_legs", "height", "weight"]:
        df = remove_outliers_iqr(df, col)

    return df


def label_data(df: pl.DataFrame):
    """
    To classify this data we are going to use Clustering. This is unsupervised learning.
    Then, the model will be saved into a pickle file.
    """

    X = df.select(
        ["walks_on_n_legs", "height", "weight", "has_wings", "has_tail"]
    ).to_numpy()

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering with KMeans (4 animals to be classified)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Add label column to the DataFrame
    df = df.with_columns(pl.Series(name="label", values=clusters))

    return X, clusters


def validate_data(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a RandomForest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Save the model into a pickle file
    with open(f"model/RFmodel.pkl", "wb") as file:
        pk.dump(rf, file)

    y_pred = rf.predict(X_test)
    print("Classification Report", classification_report(y_test, y_pred))


if __name__ == "__main__":
    data_points = 3000
    get_data(data_points)
    df = create_dataframe()
    print(df)
    #date_id = datetime.now().isoformat()
    X, y = label_data(df)
    validate_data(X, y)
