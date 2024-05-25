import glob
import os
import re
from datetime import datetime
from io import StringIO

import httpx
import pandas as pd
import torch
from httpx import URL

from config.polish_name import PolishNameConfig


class PolishNameDataLoader:
    def __init__(self, polish_name_config: PolishNameConfig):
        self.config = polish_name_config

    @staticmethod
    def download_data(url) -> pd.DataFrame:
        resource = httpx.get(url, follow_redirects=True)
        data = resource.content.decode("utf-8")
        data = pd.read_csv(
            StringIO(data),
            dtype={"name": str, "sex": str, "occurrences": int},
            header=0,
            names=["name", "sex", "occurrences"],
        )
        data = data[data["name"].str.contains(r"[a-żA-Ż]$", regex=True, na=False)]
        data["sex"] = data["sex"].replace({"MĘŻCZYZNA": "male", "KOBIETA": "female"})

        return data

    def save_data(self, data: pd.DataFrame, file_type, time):
        directory = "latest_resources"
        self.ensure_directory_exists(directory)
        data.to_csv(f"./{directory}/{file_type}_{time}.csv", index=False, header=True)

    @staticmethod
    def get_data(url) -> dict | None:
        response = httpx.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print("API communication error: ", response.status_code)
            return None

    @staticmethod
    def process_data(data: dict) -> (URL, datetime, URL, datetime):
        resources = data.get("data", [])

        latest_male_resource = None
        latest_female_resource = None
        latest_male_time = None
        latest_female_time = None

        for resource in resources:
            resource_name = resource.get("attributes", {}).get("title", "").lower()
            resource_time = resource.get("attributes", {}).get("created", "")
            if resource_time:
                resource_time = datetime.strptime(resource_time, "%Y-%m-%dT%H:%M:%SZ")

                if "male" in resource_name or "męskich" in resource_name:
                    if latest_male_time is None or resource_time > latest_male_time:
                        latest_male_time = resource_time
                        latest_male_resource = resource

                elif "female" in resource_name or "żeńskich" in resource_name:
                    if latest_female_time is None or resource_time > latest_female_time:
                        latest_female_time = resource_time
                        latest_female_resource = resource

        return (
            URL(latest_male_resource["attributes"]["csv_file_url"]),
            latest_male_time,
            URL(latest_female_resource["attributes"]["csv_file_url"]),
            latest_female_time,
        )

    def refresh_data(self):
        response = self.get_data(str(self.config.resource))

        if response is not None:
            (
                latest_male_resource,
                latest_male_time,
                latest_female_resource,
                latest_female_time,
            ) = self.process_data(response)

            if latest_male_resource:
                data = self.download_data(latest_male_resource)
                self.save_data(data, "male", latest_male_time)

            if latest_female_resource:
                data = self.download_data(latest_female_resource)
                self.save_data(data, "female", latest_female_time)

            print("Data downloaded successfully.")
        else:
            print("API communication error")

    @staticmethod
    def ensure_directory_exists(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def load_data(self) -> (pd.DataFrame, pd.DataFrame):
        print("Loading data...")
        male_file, female_file = self.get_latest_files()

        male_data, female_data = None, None
        if male_file:
            male_data = pd.read_csv(
                male_file,
                dtype={"name": str, "sex": str, "occurrences": int},
                header=0,
                names=["name", "sex", "occurrences"],
            )
            male_data = male_data[
                male_data.name.str.contains(r"[a-żA-Ż]$", regex=True, na=False)
            ]
            # male_data = male_data[male_data.occurrences > 1000]

        if female_file:
            female_data = pd.read_csv(
                female_file,
                dtype={"name": str, "sex": str, "occurrences": int},
                header=0,
                names=["name", "sex", "occurrences"],
            )
            female_data = female_data[
                female_data.name.str.contains(r"[a-żA-Ż]$", regex=True, na=False)
            ]
            female_data = female_data[female_data.occurrences > 1000]

        return male_data, female_data

    @staticmethod
    def get_timestamp_from_filename(filename):
        match = re.search(r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}", filename)
        if match:
            timestamp_str = match.group().replace("%", ":")
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        else:
            return None

    def get_latest_files(self):
        male_files = glob.glob("./latest_resources/male*.csv")
        female_files = glob.glob("./latest_resources/female*.csv")

        male_files.sort(key=self.get_timestamp_from_filename)
        female_files.sort(key=self.get_timestamp_from_filename)

        latest_male_file = male_files[-1] if male_files else None
        latest_female_file = female_files[-1] if female_files else None

        return latest_male_file, latest_female_file
