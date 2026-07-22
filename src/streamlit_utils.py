import streamlit as st
from pathlib import Path
import hashlib

from src.worldcup_predictor import WorldCupPredictor


def _get_model_version():
    """Get a hash of the model files to invalidate cache when model is retrained."""
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "models" / "worldcup_predictor"

    if not model_dir.exists():
        return None

    # Hash the evaluation_results.json to detect model changes
    eval_results_file = model_dir / "evaluation_results.json"
    if eval_results_file.exists():
        with open(eval_results_file, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    return None


@st.cache_resource
def load_predictor(_model_version=None):
    # The _model_version parameter is used to invalidate cache when model changes

    predictor = WorldCupPredictor()

    # Load trained model
    predictor.load_saved_model()

    # Base directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Load datasets needed for feature generation
    predictor.load_data(
        BASE_DIR / "data/raw/matches.csv",
        BASE_DIR / "data/raw/teams.csv",
        BASE_DIR / "data/raw/tournaments.csv"
    )

    predictor.clean_data()

    predictor.standardize_team_names()

    # Load 2026 groups
    predictor.load_2026_teams()

    return predictor