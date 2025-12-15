"Tiny helpers for saving and loading a project file in JSON"
"Keeps I/O in one place so the rest of the app can stay focused on UI/logic."

import json

def load_project(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)
  
def save_project(project: dict, path: str) -> None:
  with open(path, "w", encoding="utf-8") as f:
    json.dump(project, f, indent=2)