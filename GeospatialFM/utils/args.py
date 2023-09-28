import yaml

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)