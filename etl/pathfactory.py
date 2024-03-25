from pathlib import Path

class PathFactory:
    def __init__(self, dataset_name:str, task_name:str):
        self.path = Path.home() / "datasets" / dataset_name
        self.shards_path = Path.home() / "shards" / dataset_name

        self.url = f"s3://datasets/{dataset_name}"
        self.shards_url = f"s3://shards/{dataset_name}"

        self.experiments_path = Path.home() / "experiments" / f"{dataset_name}_{task_name}"
        self.experiments_url = f"s3://experiments/{dataset_name}_{task_name}"

    def print_paths(self):
        print(f"Local Dataset (.path): {self.path}")
        print(f"Local Shards (.shards_path): {self.shards_path}")
        print(f"Remote Dataset (.url): {self.url}")
        print(f"Remote Shards (.shards_url): {self.shards_url}")
        print(f"Local Experiments (.experiments_path): {self.experiments_path}")
        print(f"Remote Experiments (.experiments_url): {self.experiments_url}")
        
    #TODO: add functions to verify, report and create local directories
    #TODO: add function to check the exisitence of bucket in s3
