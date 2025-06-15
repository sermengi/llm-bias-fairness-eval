from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

CONFIG_FILE_PATH = CONFIGS_DIR / "config.yaml"
CONTEXT_CONFIG_FILE_PATH = CONFIGS_DIR / "context_templates.yaml"
