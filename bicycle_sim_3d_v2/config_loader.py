"""Small YAML config loader. Requires PyYAML (`pip install pyyaml`)."""
import os
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(relative_path):
    """relative_path is resolved relative to this file's directory
    (e.g. 'config/sim_params.yaml'), not the caller's cwd -- so this
    works the same regardless of where you run main.py from.
    """
    path = os.path.join(_HERE, relative_path)
    with open(path, 'r') as f:
        return yaml.safe_load(f)