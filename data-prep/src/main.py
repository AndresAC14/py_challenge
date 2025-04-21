import pandas as pd
import requests
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 

def get_data() -> None:
    '''
    Get the schema and the full data and save them into two separate JSON files.
    '''

    schema_response = requests.get("http://127.0.0.1:8777/api/v1/animals/schema")
    data_response = requests.post("http://127.0.0.1:8777/api/v1/animals/data", json={
        "seed": 42,
        "number_of_datapoints": 1000
    })

    # Check for request success
    if schema_response.status_code == 200 and data_response.status_code == 200:
        with open("data/schema.json", "w") as schema_out:
            json.dump(schema_response.json(), schema_out, indent=2)

        with open("data/animals.json", "w") as data_out:
            json.dump(data_response.json(), data_out, indent=2)
    else:
        raise Exception("Failed to fetch data from the API")


def create_dataframe() -> pd.DataFrame:
    '''
    Transforms the JSON data into a Pandas DataFrame.
    '''
    with open("data/animals.json", "r") as data_file:
        data = json.load(data_file)

    df = pd.DataFrame(data)

    return df


def label_data(df: pd.DataFrame, n_clusters=4) -> pd.DataFrame:
    '''
    To classify this data we are going to use Clustering. This is unsupervised learning.
    '''

    features = df.columns
    
    # Convert booleans to integers
    df_encoded = df.copy()
    df_encoded["has_wings"] = df_encoded["has_wings"].astype(int)
    df_encoded["has_tail"] = df_encoded["has_tail"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    print(df.head(5))

    return df


if __name__ == "__main__":
    #get_data()
    df = create_dataframe()
    # label_data(df)
    
    