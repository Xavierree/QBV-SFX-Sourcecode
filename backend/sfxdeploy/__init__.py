from .config import setup_env
setup_env()          # HF_HOME inside the repo + offline + vendor/ on sys.path, before anything imports transformers
__version__ = '0.1.0'
