#1. Import the libraries, classes, modules needed for the project

import pandas as pd
import numpy as np
from copy import deepcopy
from collections import defaultdict
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_score,recall_score,
    f1_score,confusion_matrix,classification_report)
import os
import json
import pickle
from itertools import combinations
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

# 2. Create the Class to have the whole project start here
class WorldCupPredictor:

    def __init__(self):
        self.df_matches = None
        self.df_teams = None
        self.df_tournaments = None
        self.training_dataset = None
        self.model = None
        self.feature_columns = None
        self.current_elo = {}
        self.team_history = {}
        self.h2h_history = {}
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_years = None
        self.test_years = None
        self.training_history = None
        self.y_pred = None
        self.y_pred_proba = None
        self.evaluation_results = None
        self.feature_importance_df = None
        self.top_features = None
        self.team_snapshots = {}
        self.team_snapshots = {}
        self.last_prediction = None
        self.teams_2026 = None
        self.groups = None
        self.group_tables = None
        self.knockout_bracket = None
        self.stage_weights = {
            "group": 1,
            "round_of_32": 2,
            "round_of_16": 3,
            "quarter_final": 4,
            "semi_final": 5,
            "third_place": 5,
            "final": 6
        }
        self.groups = {}

        self.group_matches = {}

        self.group_tables = {}

        self.group_match_results = {}

        self.third_place_table = None

        self.round_of_32 = []

        self.round_of_16 = []

        self.quarter_finals = []

        self.semi_finals = []

        self.final = None

        self.third_place_match = None

        self.fifa_rankings = {}

    # 1. Load the data and confirm
    def load_data(self,matches_path,teams_path,tournaments_path):
        if isinstance(matches_path, pd.DataFrame):
            self.df_matches = matches_path
        else:
            self.df_matches = pd.read_csv(matches_path)

        if isinstance(teams_path, pd.DataFrame):
            self.df_teams = teams_path
        else:
            self.df_teams = pd.read_csv(teams_path)

        if isinstance(tournaments_path, pd.DataFrame):
            self.df_tournaments = tournaments_path
        else:
            self.df_tournaments = pd.read_csv(tournaments_path)


        print("Data loaded successfully")
        print(f"Matches: {self.df_matches.shape}")
        print(f"Teams: {self.df_teams.shape}")
        print(f"Tournaments: {self.df_tournaments.shape}")

    # 4. Clean the data, remove spaces, set the right date formats, set all integers to floats
    def clean_data(self):
        # Dates
        self.df_matches["date"] = pd.to_datetime(
            self.df_matches["date"],
            errors="coerce"
        )

        # Scores
        self.df_matches["homeScore"] = pd.to_numeric(
            self.df_matches["homeScore"],
            errors="coerce"
        )
        self.df_matches["awayScore"] = pd.to_numeric(
            self.df_matches["awayScore"],
            errors="coerce"
        )

        # Strip spaces
        string_columns = ["homeTeam", "awayTeam", "stage", "country"]
        for col in string_columns:
            self.df_matches[col] = (
                self.df_matches[col]
                .astype(str)
                .str.strip()
            )

        self.df_teams["name"] = (
            self.df_teams["name"]
            .astype(str)
            .str.strip()
        )
        self.df_teams["confederation"] = (
            self.df_teams["confederation"]
            .astype(str)
            .str.strip()
        )

        # Fill missing scores
        self.df_matches["homeScore"] = self.df_matches["homeScore"].fillna(0)
        self.df_matches["awayScore"] = self.df_matches["awayScore"].fillna(0)

        print("Cleaning completed")

    # 5. Remove the matches from 2026 because they are what we are predicting
    def remove_future_matches(self, prediction_year=2026):
        before = len(self.df_matches)
        self.df_matches = self.df_matches[
            self.df_matches["tournament_year"] < prediction_year
        ].copy()
        after = len(self.df_matches)

        print(f"Removed {before-after} future matches")
        print(f"Remaining matches: {after}")

    # 6. Rename some fields due to changes in country names in the world cup history
    def standardize_team_names(self):
        mapping = {
            "West Germany": "Germany",
            "FR Yugoslavia": "Serbia",
            "Serbia and Montenegro": "Serbia",
            "Soviet Union": "Russia",
            "Czechoslovakia": "Czech Republic"
        }

        self.df_matches["homeTeam"] = self.df_matches["homeTeam"].replace(mapping)
        self.df_matches["awayTeam"] = self.df_matches["awayTeam"].replace(mapping)
        self.df_teams["name"] = self.df_teams["name"].replace(mapping)

        print("Team names standardized")

    # Check the data that its loaded, cleaned and formatted to the right types, renamed fields to capture change of country names
    def validate_data(self):
        print("="*50)
        print("MATCHES")
        print(self.df_matches.shape)
        print(self.df_matches.isnull().sum())

        print("="*50)
        print("TEAMS")
        print(self.df_teams.shape)
        print(self.df_teams.isnull().sum())

        print("="*50)
        print("TOURNAMENTS")
        print(self.df_tournaments.shape)
        print(self.df_tournaments.isnull().sum())

    def normalize_stages(self):
        # Creates ML-friendly stage features while preserving the original tournament structure
        stage_map = {
            # Group Stage
            "group_1": "group",
            "group_2": "group",
            "group_3": "group",
            "group_4": "group",
            "group_5": "group",
            "group_6": "group",
            "group_7": "group",
            "group_8": "group",

            # Knockout
            "r32": "round_of_32",
            "r16": "round_of_16",
            "qf": "quarter_final",
            "sf": "semi_final",
            "final": "final",
            "third_place": "third_place"
        }

        # Preserve original stage
        self.df_matches["stage_category"] = (
            self.df_matches["stage"]
            .astype(str)
            .str.lower()
            .map(stage_map)
        )

        # Extract group number
        self.df_matches["group_number"] = (
            self.df_matches["stage"]
            .astype(str)
            .str.extract(r'group_(\d+)')[0]
        )

        print("Stage normalization completed.")
        print(self.df_matches[["stage", "stage_category", "group_number"]].head(20))

    # Build  target and separate home and away
    def build_target(self):
        """
        Creates target variable.
        Home Win = 1
        Draw = 0
        Away Win = -1
        """

        self.df_matches["target"] = np.where(
            self.df_matches["homeScore"] > self.df_matches["awayScore"],
            1,
            np.where(
                self.df_matches["homeScore"] < self.df_matches["awayScore"],
                -1,
                0
            )
        )

        print("Target variable created.")
        print(
            self.df_matches[
                [
                    "homeTeam",
                    "awayTeam",
                    "homeScore",
                    "awayScore",
                    "target"
                ]
            ].head(10)
        )

    #sort the matches in ascending order to help calculate for rolling features
    def sort_matches(self):

        self.df_matches = (
            self.df_matches
            .sort_values(by=["date"])
            .reset_index(drop=True)
        )

        print("Matches sorted chronologically.")
        print(
            self.df_matches[
                [
                    "date",
                    "homeTeam",
                    "awayTeam"
                ]
            ].head()
        )

    # Calculate the Elo Ratings
    def calculate_elo(self, k_factor=30, initial_elo=1500):
        """
        Calculate pre-match Elo ratings.

        World Cup matches are treated as neutral venue games.

        Features created:
        -----------------
        home_team_elo
        away_team_elo
        elo_difference
        """

        elo = {}

        home_elos = []
        away_elos = []
        elo_differences = []

        for _, match in self.df_matches.iterrows():
            team1 = match["homeTeam"]
            team2 = match["awayTeam"]

            # Initialize ratings
            if team1 not in elo:
                elo[team1] = initial_elo

            if team2 not in elo:
                elo[team2] = initial_elo

            # Store ratings BEFORE match
            team1_elo = elo[team1]
            team2_elo = elo[team2]

            home_elos.append(team1_elo)
            away_elos.append(team2_elo)
            elo_differences.append(team1_elo - team2_elo)

            # Expected result
            expected_team1 = 1 / (
                1 + 10 ** ((team2_elo - team1_elo) / 400)
            )
            expected_team2 = 1 - expected_team1

            # Actual result
            if match["homeScore"] > match["awayScore"]:
                actual_team1 = 1
                actual_team2 = 0
            elif match["homeScore"] < match["awayScore"]:
                actual_team1 = 0
                actual_team2 = 1
            else:
                actual_team1 = 0.5
                actual_team2 = 0.5

            # Update Elo
            elo[team1] = (
                team1_elo
                + k_factor * (actual_team1 - expected_team1)
            )
            elo[team2] = (
                team2_elo
                + k_factor * (actual_team2 - expected_team2)
            )

        # Save features (unindented from the for loop, aligned with the method body)
        self.df_matches["home_team_elo"] = home_elos
        self.df_matches["away_team_elo"] = away_elos
        self.df_matches["elo_difference"] = elo_differences

        # Save latest ratings for tournament simulation
        self.current_elo = elo

        print("Elo ratings calculated.")
        print(
            self.df_matches[
                [
                    "homeTeam",
                    "awayTeam",
                    "home_team_elo",
                    "away_team_elo",
                    "elo_difference"
                ]
            ].head(10)
        )
        self.current_elo = elo

    # helper function retrieving the current Elo rating
    def get_current_elo(self, team):

        return self.current_elo.get(team, 1500)

    # the recent 5 match results 'recent matches team form'
    def build_rolling_features(self, window=5):
        """
        Build rolling World Cup form features.

        Uses only matches BEFORE the current match.
        Ignores API home/away designation.
        """

        from collections import defaultdict

        team_history = defaultdict(list)

        home_win_rate = []
        away_win_rate = []
        home_goals = []
        away_goals = []
        home_conceded = []
        away_conceded = []
        home_goal_diff = []
        away_goal_diff = []

        for _, match in self.df_matches.iterrows():
            team1 = match["homeTeam"]
            team2 = match["awayTeam"]

            # ---------------------
            # TEAM 1 FEATURES
            # ---------------------
            recent = team_history[team1][-window:]

            if len(recent) == 0:
                home_win_rate.append(0)
                home_goals.append(0)
                home_conceded.append(0)
                home_goal_diff.append(0)
            else:
                home_win_rate.append(
                    np.mean([x["result"] for x in recent])
                )
                home_goals.append(
                    np.mean([x["gf"] for x in recent])
                )
                home_conceded.append(
                    np.mean([x["ga"] for x in recent])
                )
                home_goal_diff.append(
                    np.mean([x["gf"] - x["ga"] for x in recent])
                )

            # ---------------------
            # TEAM 2 FEATURES
            # ---------------------
            recent = team_history[team2][-window:]

            if len(recent) == 0:
                away_win_rate.append(0)
                away_goals.append(0)
                away_conceded.append(0)
                away_goal_diff.append(0)
            else:
                away_win_rate.append(
                    np.mean([x["result"] for x in recent])
                )
                away_goals.append(
                    np.mean([x["gf"] for x in recent])
                )
                away_conceded.append(
                    np.mean([x["ga"] for x in recent])
                )
                away_goal_diff.append(
                    np.mean([x["gf"] - x["ga"] for x in recent])
                )

            # ---------------------
            # UPDATE TEAM HISTORY
            # ---------------------
            hs = match["homeScore"]
            aw = match["awayScore"]

            if hs > aw:
                team1_result = 1
                team2_result = 0
            elif hs < aw:
                team1_result = 0
                team2_result = 1
            else:
                team1_result = 0.5
                team2_result = 0.5

            team_history[team1].append({
                "result": team1_result,
                "gf": hs,
                "ga": aw
            })

            team_history[team2].append({
                "result": team2_result,
                "gf": aw,
                "ga": hs
            })

        # Save features (Outside the for-loop)
        self.df_matches["home_win_rate_last_5"] = home_win_rate
        self.df_matches["away_win_rate_last_5"] = away_win_rate
        self.df_matches["home_goals_last_5"] = home_goals
        self.df_matches["away_goals_last_5"] = away_goals
        self.df_matches["home_conceded_last_5"] = home_conceded
        self.df_matches["away_conceded_last_5"] = away_conceded
        self.df_matches["home_goal_diff_last_5"] = home_goal_diff
        self.df_matches["away_goal_diff_last_5"] = away_goal_diff

        # Save latest history mapping for simulation states
        self.team_history = team_history

        print("Rolling features calculated.")
        print(
            self.df_matches[
                [
                    "homeTeam",
                    "awayTeam",
                    "home_win_rate_last_5",
                    "away_win_rate_last_5",
                    "home_goals_last_5",
                    "away_goals_last_5"
                ]
            ].head(10)
        )

    # build a head to head team pairings
    def build_head_to_head(self):
        h2h_history = defaultdict(list)

        h2h_games = []
        h2h_win_rate = []
        h2h_goal_diff = []

        for _, match in self.df_matches.iterrows():
            team1 = match["homeTeam"]
            team2 = match["awayTeam"]

            pair = tuple(sorted([team1, team2]))
            previous = h2h_history[pair]

            # -----------------------
            # BUILD FEATURES
            # -----------------------
            if len(previous) == 0:
                h2h_games.append(0)
                h2h_win_rate.append(0.5)
                h2h_goal_diff.append(0)
            else:
                h2h_games.append(len(previous))
                h2h_win_rate.append(
                    np.mean([x["result"] for x in previous])
                )
                h2h_goal_diff.append(
                    np.sum([x["goal_diff"] for x in previous])
                )

            # -----------------------
            # UPDATE HISTORY
            # -----------------------
            hs = match["homeScore"]
            aw = match["awayScore"]

            if pair[0] == team1:
                goal_diff = hs - aw
                if hs > aw:
                    result = 1
                elif hs < aw:
                    result = 0
                else:
                    result = 0.5
            else:
                goal_diff = aw - hs
                if aw > hs:
                    result = 1
                elif aw < hs:
                    result = 0
                else:
                    result = 0.5

            h2h_history[pair].append({
                "result": result,
                "goal_diff": goal_diff
            })

        # Save features (Outside the for-loop)
        self.df_matches["head_to_head_games"] = h2h_games
        self.df_matches["head_to_head_win_rate"] = h2h_win_rate
        self.df_matches["head_to_head_goal_difference"] = h2h_goal_diff

        # Save historical lookup dictionary for out-of-sample simulation metrics
        self.h2h_history = h2h_history

        print("Head-to-head features calculated.")
        print(
            self.df_matches[
                [
                    "homeTeam",
                    "awayTeam",
                    "head_to_head_games",
                    "head_to_head_win_rate",
                    "head_to_head_goal_difference"
                ]
            ].head(10)
        )
    # Build the historical performance rating of teams
    def build_historical_features(self):

        home_titles = []
        away_titles = []
        home_knockout = []
        away_knockout = []
        home_avg_finish = []
        away_avg_finish = []
        home_apps = []
        away_apps = []
        home_sf = []
        away_sf = []
        home_final = []
        away_final = []

        for _, match in self.df_matches.iterrows():
            year = match["tournament_year"]
            team1 = match["homeTeam"]
            team2 = match["awayTeam"]

            # ------------------------
            # TEAM 1
            # ------------------------
            history = self.df_teams[
                (self.df_teams["name"] == team1)
                & (self.df_teams["tournament_year"] < year)
            ]

            apps = len(history)
            home_apps.append(apps)

            if apps == 0:
                home_titles.append(0)
                home_knockout.append(0)
                home_avg_finish.append(32)
                home_sf.append(0)
                home_final.append(0)
            else:
                home_titles.append(
                    (history["finalPosition"] == 1).sum()
                )
                home_avg_finish.append(
                    history["finalPosition"]
                    .fillna(32)
                    .mean()
                )

                knockout_count = (
                    history["knockoutPath"]
                    .fillna("[]")
                    .astype(str) != "[]"
                ).sum()
                home_knockout.append(knockout_count / apps)

                sf_count = (
                    history["knockoutPath"]
                    .fillna("")
                    .astype(str)
                    .str.contains("'stage': 'sf'")
                ).sum()
                home_sf.append(sf_count / apps)

                final_count = (
                    history["knockoutPath"]
                    .fillna("")
                    .astype(str)
                    .str.contains("'stage': 'final'")
                ).sum()
                home_final.append(final_count / apps)

            # ------------------------
            # TEAM 2
            # ------------------------
            history = self.df_teams[
                (self.df_teams["name"] == team2)
                & (self.df_teams["tournament_year"] < year)
            ]

            apps = len(history)
            away_apps.append(apps)

            if apps == 0:
                away_titles.append(0)
                away_knockout.append(0)
                away_avg_finish.append(32)
                away_sf.append(0)
                away_final.append(0)
            else:
                away_titles.append(
                    (history["finalPosition"] == 1).sum()
                )
                away_avg_finish.append(
                    history["finalPosition"]
                    .fillna(32)
                    .mean()
                )

                knockout_count = (
                    history["knockoutPath"]
                    .fillna("[]")
                    .astype(str) != "[]"
                ).sum()
                away_knockout.append(knockout_count / apps)

                sf_count = (
                    history["knockoutPath"]
                    .fillna("")
                    .astype(str)
                    .str.contains("'stage': 'sf'")
                ).sum()
                away_sf.append(sf_count / apps)

                final_count = (
                    history["knockoutPath"]
                    .fillna("")
                    .astype(str)
                    .str.contains("'stage': 'final'")
                ).sum()
                away_final.append(final_count / apps)

        # Save features back to DataFrame (Outside the loop)
        self.df_matches["home_historical_titles"] = home_titles
        self.df_matches["away_historical_titles"] = away_titles
        self.df_matches["home_knockout_rate"] = home_knockout
        self.df_matches["away_knockout_rate"] = away_knockout
        self.df_matches["home_average_finish"] = home_avg_finish
        self.df_matches["away_average_finish"] = away_avg_finish
        self.df_matches["home_world_cup_appearances"] = home_apps
        self.df_matches["away_world_cup_appearances"] = away_apps
        self.df_matches["home_semi_final_rate"] = home_sf
        self.df_matches["away_semi_final_rate"] = away_sf
        self.df_matches["home_final_rate"] = home_final
        self.df_matches["away_final_rate"] = away_final

        print("Historical features calculated.")
        print(
            self.df_matches[
                [
                    "homeTeam",
                    "awayTeam",
                    "home_historical_titles",
                    "away_historical_titles",
                    "home_world_cup_appearances",
                    "away_world_cup_appearances",
                    "home_final_rate",
                    "away_final_rate"
                ]
            ].head(10)
        )

    # adding weight to the tournament stages
    def build_stage_weight(self):

        stage_map = {
            "group_a": 1,
            "group_b": 1,
            "group_c": 1,
            "group_d": 1,
            "group_e": 1,
            "group_f": 1,
            "group_g": 1,
            "group_h": 1,
            "round_of_16": 2,
            "quarter_final": 3,
            "semi_final": 4,
            "third_place": 4,
            "final": 5
        }

        self.df_matches["stage_weight"] = (
            self.df_matches["stageNormalized"]
            .map(stage_map)
            .fillna(1)
        )

        print("Stage weights created.")
        print(
            self.df_matches[
                [
                    "stageNormalized",
                    "stage_weight"
                ]
            ]
            .drop_duplicates()
            .sort_values("stage_weight")
        )

    # Build a training data set
    def build_training_dataset(self):
        self.feature_columns = [
            "home_team_elo",
            "away_team_elo",
            "elo_difference",

            "home_win_rate_last_5",
            "away_win_rate_last_5",

            "home_goals_last_5",
            "away_goals_last_5",

            "home_conceded_last_5",
            "away_conceded_last_5",

            "home_goal_diff_last_5",
            "away_goal_diff_last_5",

            "head_to_head_games",
            "head_to_head_win_rate",
            "head_to_head_goal_difference",

            "home_historical_titles",
            "away_historical_titles",

            "home_knockout_rate",
            "away_knockout_rate",

            "home_average_finish",
            "away_average_finish",

            "home_world_cup_appearances",
            "away_world_cup_appearances",

            "home_semi_final_rate",
            "away_semi_final_rate",

            "home_final_rate",
            "away_final_rate",

            "stage_weight"
        ]

        required_columns = (
            self.feature_columns
            + [
                "target",
                "homeTeam",
                "awayTeam",
                "date",
                "tournament_year"
            ]
        )

        self.training_dataset = (
            self.df_matches[required_columns]
            .copy()
        )

        # Fill any missing feature values
        self.training_dataset[self.feature_columns] = (
            self.training_dataset[self.feature_columns]
            .fillna(0)
        )

        # Remove rows without target
        self.training_dataset = (
            self.training_dataset[
                self.training_dataset["target"].notna()
            ]
            .reset_index(drop=True)
        )

        print("=" * 60)
        print("Training dataset created")
        print("=" * 60)

        print(f"Rows: {self.training_dataset.shape[0]}")
        print(f"Columns: {self.training_dataset.shape[1]}")
        print()

        print("Feature columns:")
        for feature in self.feature_columns:
            print(feature)

        print()
        print(self.training_dataset.head())

    #Select features and separate them from the target
    def select_features(self):
        """
        Build feature matrix X and target vector y.
        """

        self.X = self.training_dataset[self.feature_columns].copy()
        self.y = self.training_dataset["target"].copy()

        print("=" * 60)
        print("Features selected")
        print("=" * 60)

        print(f"X Shape : {self.X.shape}")
        print(f"y Shape : {self.y.shape}")
        print()

        print("Feature Columns")
        for col in self.feature_columns:
            print(f"- {col}")

        print()
        print("Target Distribution")
        print(
            self.y.value_counts()
            .sort_index()
        )
        self.n_features = len(self.feature_columns)

        return self.X, self.y

    # Split the training and testing data training 1960 - 2014 training ---2018-2022 testing
    def split_data(self, test_tournaments=2):
        """
        Time-based split.

        Default:
        ----------
        Train : All but latest 2 tournaments
        Test  : Latest 2 tournaments
        """

        years = sorted(
            self.training_dataset["tournament_year"].unique()
        )

        self.test_years = years[-test_tournaments:]
        self.train_years = years[:-test_tournaments]

        train_mask = (
            self.training_dataset["tournament_year"].isin(
                self.train_years
            )
        )

        test_mask = (
            self.training_dataset["tournament_year"].isin(
                self.test_years
            )
        )

        self.X_train = self.training_dataset.loc[
            train_mask,
            self.feature_columns
        ]

        self.X_test = self.training_dataset.loc[
            test_mask,
            self.feature_columns
        ]

        self.y_train = self.training_dataset.loc[
            train_mask,
            "target"
        ]

        self.y_test = self.training_dataset.loc[
            test_mask,
            "target"
        ]

        print("=" * 60)
        print("Time-based split completed")
        print("=" * 60)
        print()

        print(f"Training tournaments : {self.train_years}")
        print(f"Testing tournaments  : {self.test_years}")
        print()

        print(f"X_train : {self.X_train.shape}")
        print(f"X_test  : {self.X_test.shape}")
        print()

        print(f"y_train : {self.y_train.shape}")
        print(f"y_test  : {self.y_test.shape}")

    #train the model using X_train, y_train
    def train_model(
        self,
        iterations=500,
        learning_rate=0.05,
        depth=6,
        early_stopping_rounds=50,
        random_seed=42
    ):
        """
        Train the World Cup prediction model.

        Stores:
            self.model
            self.class_labels
        """

        # -----------------------------
        # Validation
        # -----------------------------
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Training data not found. Run split_data() first."
            )

        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Test data not found. Run split_data() first."
            )

        # -----------------------------
        # Initialize model
        # -----------------------------
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            random_seed=random_seed,
            verbose=100
        )

        # -----------------------------
        # Train model
        # -----------------------------
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=(
                self.X_test,
                self.y_test
            ),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds
        )

        # -----------------------------
        # Store model
        # -----------------------------
        self.model = model
        self.class_labels = model.classes_
        self.training_history = model.get_evals_result()

        # -----------------------------
        # Training summary
        # -----------------------------
        print("=" * 50)
        print("Model training completed.")
        print("=" * 50)

        print(f"Best Iteration : {model.get_best_iteration()}")
        print(f"Best Score     : {model.get_best_score()}")

        print("\nClass Labels:")
        print(self.class_labels)


        return self.model

    # Evaluate the model metrics
    def evaluate_model(self):
        """
        Evaluate the trained model.

        Stores:
            self.y_pred
            self.y_pred_proba
            self.evaluation_results
        """

        # --------------------------------
        # Validation
        # --------------------------------

        if self.model is None:
            raise ValueError(
                "Model not trained. Run train_model() first."
            )

        # --------------------------------
        # Predictions
        # --------------------------------

        y_pred = self.model.predict(self.X_test)

        # CatBoost may return shape (n,1)
        y_pred = y_pred.flatten()

        y_pred_proba = self.model.predict_proba(self.X_test)

        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        # --------------------------------
        # Metrics
        # --------------------------------

        accuracy = accuracy_score(
            self.y_test,
            y_pred
        )

        balanced_accuracy = balanced_accuracy_score(
            self.y_test,
            y_pred
        )

        precision = precision_score(
            self.y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            self.y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            self.y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        cm = confusion_matrix(
            self.y_test,
            y_pred,
            labels=self.class_labels
        )

        report = classification_report(
            self.y_test,
            y_pred,
            labels=self.class_labels,
            zero_division=0
        )

        # --------------------------------
        # Store results
        # --------------------------------

        self.evaluation_results = {

            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report

        }

        # --------------------------------
        # Print Summary
        # --------------------------------

        print("=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
        print(f"Precision (Macro):  {precision:.4f}")
        print(f"Recall (Macro):     {recall:.4f}")
        print(f"F1 Score (Macro):   {f1:.4f}")

        print("\nConfusion Matrix")
        print(cm)

        print("\nClassification Report")
        print(report)

        return self.evaluation_results

    # Feature importance
    def feature_importance(self):
        """
        Calculate and store feature importance.

        Stores:
            self.feature_importance_df
        """

        # --------------------------------
        # Validation
        # --------------------------------

        if self.model is None:
            raise ValueError(
                "Model not trained. Run train_model() first."
            )

        # --------------------------------
        # Calculate importance
        # --------------------------------

        importance = self.model.get_feature_importance()

        importance_df = pd.DataFrame({

            "feature": self.feature_columns,
            "importance": importance

        })

        importance_df = (
            importance_df
            .sort_values(
                by="importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        # --------------------------------
        # Store
        # --------------------------------

        self.feature_importance_df = importance_df
        self.top_features = importance_df.head(10)

        # --------------------------------
        # Display
        # --------------------------------

        print("=" * 50)
        print("FEATURE IMPORTANCE")
        print("=" * 50)

        print(importance_df)

        return importance_df

    # save the model to be accessed via an api
    def save_model(self, save_dir="worldcup_predictor"):
        """
        Save the trained model and supporting artifacts.

        Saves:
            - CatBoost model
            - Feature columns
            - Class labels
            - Current Elo ratings
            - Team history
            - Head-to-head history
            - Metadata
        """

        # --------------------------------
        # Validation
        # --------------------------------
        if self.model is None:
            raise ValueError(
                "No trained model found. Run train_model() first."
            )

        # --------------------------------
        # Create directory
        # --------------------------------
        os.makedirs(save_dir, exist_ok=True)

        # --------------------------------
        # Save CatBoost model
        # --------------------------------
        self.model.save_model(
            os.path.join(
                save_dir,
                "catboost_model.cbm"
            )
        )

        # --------------------------------
        # Save supporting objects
        # --------------------------------
        with open(
            os.path.join(
                save_dir,
                "feature_columns.pkl"
            ),
            "wb"
        ) as f:
            pickle.dump(self.feature_columns, f)

        with open(
            os.path.join(
                save_dir,
                "class_labels.pkl"
            ),
            "wb"
        ) as f:
            pickle.dump(self.class_labels, f)

        with open(
            os.path.join(
                save_dir,
                "current_elo.pkl"
            ),
            "wb"
        ) as f:
            pickle.dump(self.current_elo, f)

        with open(
            os.path.join(
                save_dir,
                "team_history.pkl"
            ),
            "wb"
        ) as f:
            pickle.dump(self.team_history, f)

        with open(
            os.path.join(
                save_dir,
                "h2h_history.pkl"
            ),
            "wb"
        ) as f:
            pickle.dump(self.h2h_history, f)

        with open(
            os.path.join(
                save_dir,
                "evaluation_results.json"
            ),
            "w"
        ) as f:
            json.dump(self.evaluation_results, f, indent=4)

        if self.feature_importance_df is not None:
            self.feature_importance_df.to_csv(
                os.path.join(
                    save_dir,
                    "feature_importance.csv"
                ),
                index=False
            )

        # --------------------------------
        # Metadata
        # --------------------------------
        metadata = {

            "train_years": [
                int(year)
                for year in self.train_years
            ],

            "test_years": [
                int(year)
                for year in self.test_years
            ],

            "n_features": int(
                len(self.feature_columns)
            ),

            "feature_names": list(
                self.feature_columns
            ),

            "model_type": "CatBoostClassifier",

            "target_mapping": {

                "-1": "Team B Win",

                "0": "Draw",

                "1": "Team A Win"

            }

        }

        with open(
            os.path.join(
                save_dir,
                "metadata.json"
            ),
            "w"
        ) as f:
            json.dump(metadata, f, indent=4)

        print("=" * 50)
        print("Model successfully saved.")
        print("=" * 50)
        print(f"Location: {save_dir}")

        return save_dir

    # Get a snapshot of the team to predict
    def get_team_snapshot(self, team_name):
        """
        Build the latest snapshot for a team.
        """

        # --------------------------
        # Elo
        # --------------------------

        if hasattr(self, "team_snapshots"):

            if team_name in self.team_snapshots:
                return self.team_snapshots[team_name]

        elo = self.current_elo.get(
            team_name,
            1500
        )

        # --------------------------
        # Rolling Form
        # --------------------------

        matches = self.team_history.get(
            team_name,
            []
        )

        recent = matches[-5:]

        if len(recent) == 0:

            win_rate = 0.5
            goals = 1.0
            conceded = 1.0
            goal_diff = 0.0

        else:

            win_rate = sum(
                x["result"]
                for x in recent
            ) / len(recent)

            goals = sum(
                x["gf"]
                for x in recent
            ) / len(recent)

            conceded = sum(
                x["ga"]
                for x in recent
            ) / len(recent)

            goal_diff = goals - conceded

        # --------------------------
        # Historical Tournament Data
        # --------------------------

        team_df = self.df_teams[
            self.df_teams["name"] == team_name
        ]

        if len(team_df) == 0:

            titles = 0
            knockout_rate = 0
            average_finish = 32
            appearances = 0
            semi_final_rate = 0
            final_rate = 0

        else:

            appearances = len(team_df)

            titles = int(
                (team_df["finalPosition"] == 1).sum()
            )

            knockout_rate = float(
                (team_df["finalPosition"] <= 16).mean()
            )

            average_finish = float(
                team_df["finalPosition"].mean()
            )

            semi_final_rate = float(
                (team_df["finalPosition"] <= 4).mean()
            )

            final_rate = float(
                (team_df["finalPosition"] <= 2).mean()
            )

        # --------------------------
        # Final Snapshot
        # --------------------------

        snapshot = {

            "elo": elo,

            "win_rate_last_5": win_rate,

            "goals_last_5": goals,

            "conceded_last_5": conceded,

            "goal_diff_last_5": goal_diff,

            "historical_titles": titles,

            "knockout_rate": knockout_rate,

            "average_finish": average_finish,

            "world_cup_appearances": appearances,

            "semi_final_rate": semi_final_rate,

            "final_rate": final_rate

        }
        self.team_snapshots[team_name] = snapshot

        return snapshot

    # get match features for predicting from a new sample
    def build_match_features(
        self,
        team_a,
        team_b,
        stage_weight=1
    ):
        """
        Build model features for a future match.

        Parameters
        ----------
        team_a : str

        team_b : str

        stage_weight : int
            Group = 1
            R16   = 2
            QF    = 3
            SF    = 4
            Third = 4
            Final = 5

        Returns
        -------
        pd.DataFrame
        """

        # --------------------------------
        # Team snapshots
        # --------------------------------

        home = self.get_team_snapshot(
            team_a
        )

        away = self.get_team_snapshot(
            team_b
        )

        # --------------------------------
        # Head-to-head
        # --------------------------------

        pair = tuple(
            sorted(
                [team_a, team_b]
            )
        )

        h2h_matches = self.h2h_history.get(
            pair,
            []
        )

        if len(h2h_matches) == 0:

            h2h_games = 0
            h2h_win_rate = 0.5
            h2h_goal_diff = 0.0

        else:

            h2h_games = len(
                h2h_matches
            )

            h2h_win_rate = sum(

                x["result"]

                for x in h2h_matches

            ) / h2h_games

            h2h_goal_diff = sum(x["goal_diff"] for x in h2h_matches) / h2h_games

        # --------------------------------
        # Build feature row
        # --------------------------------

        row = {

            "home_team_elo":
                home["elo"],

            "away_team_elo":
                away["elo"],

            "elo_difference":
                home["elo"] - away["elo"],

            "home_win_rate_last_5":
                home["win_rate_last_5"],

            "away_win_rate_last_5":
                away["win_rate_last_5"],

            "home_goals_last_5":
                home["goals_last_5"],

            "away_goals_last_5":
                away["goals_last_5"],

            "home_conceded_last_5":
                home["conceded_last_5"],

            "away_conceded_last_5":
                away["conceded_last_5"],

            "home_goal_diff_last_5":
                home["goal_diff_last_5"],

            "away_goal_diff_last_5":
                away["goal_diff_last_5"],

            "head_to_head_games":
                h2h_games,

            "head_to_head_win_rate":
                h2h_win_rate,

            "head_to_head_goal_difference":
                h2h_goal_diff,

            "home_historical_titles":
                home["historical_titles"],

            "away_historical_titles":
                away["historical_titles"],

            "home_knockout_rate":
                home["knockout_rate"],

            "away_knockout_rate":
                away["knockout_rate"],

            "home_average_finish":
                home["average_finish"],

            "away_average_finish":
                away["average_finish"],

            "home_world_cup_appearances":
                home["world_cup_appearances"],

            "away_world_cup_appearances":
                away["world_cup_appearances"],

            "home_semi_final_rate":
                home["semi_final_rate"],

            "away_semi_final_rate":
                away["semi_final_rate"],

            "home_final_rate":
                home["final_rate"],

            "away_final_rate":
                away["final_rate"],

            "stage_weight":
                stage_weight

        }

        # --------------------------------
        # Create DataFrame
        # --------------------------------

        features = pd.DataFrame(
            [row]
        )

        # --------------------------------
        # Ensure exact order
        # --------------------------------

        features = features[
            self.feature_columns
        ]
        missing = set(
            self.feature_columns
        ) - set(
            features.columns
        )

        if missing:

            raise ValueError(
                f"Missing features: {missing}"
            )

        features = features[
            self.feature_columns
        ]

        return features

    # Predict match
    def predict_match(
        self,
        team_a,
        team_b,
        stage_weight=1
    ):
        """
        Predict the outcome of a match.

        Parameters
        ----------
        team_a : str

        team_b : str

        stage_weight : int

        Returns
        -------
        dict
        """

        # -----------------------------
        # Validation
        # -----------------------------

        if self.model is None:
            raise ValueError(
                "Model not trained. Run train_model() first."
            )

        # -----------------------------
        # Build features
        # -----------------------------

        features = self.build_match_features(

            team_a,

            team_b,

            stage_weight

        )

        # -----------------------------
        # Predict
        # -----------------------------

        prediction = self.model.predict(
            features
        )

        # CatBoost returns [[1]]
        prediction = int(
            prediction.flatten()[0]
        )

        # -----------------------------
        # Human-readable label
        # -----------------------------

        if prediction == 1:

            result = f"{team_a} Win"

        elif prediction == -1:

            result = f"{team_b} Win"

        else:

            result = "Draw"

        # -----------------------------
        # Output
        # -----------------------------

        output = {

            "team_a": team_a,

            "team_b": team_b,

            "prediction": prediction,

            "result": result

        }
        self.last_prediction = output
        return output

    # predict match probabilities
    def predict_match_proba(
        self,
        team_a,
        team_b,
        stage_weight=1
    ):
        """
        Predict match probabilities.

        Returns
        -------
        dict
        """

        # ----------------------------
        # Validation
        # ----------------------------

        if self.model is None:
            raise ValueError(
                "Model not trained."
            )

        # ----------------------------
        # Build features
        # ----------------------------

        features = self.build_match_features(

            team_a,

            team_b,

            stage_weight

        )

        # ----------------------------
        # Predict probabilities
        # ----------------------------

        probabilities = self.model.predict_proba(
            features
        )[0]

        # ----------------------------
        # Build result
        # ----------------------------

        output = {

            "team_a": team_a,

            "team_b": team_b,

            "probabilities": {}

        }

        for label, probability in zip(

            self.class_labels,

            probabilities

        ):

            output["probabilities"][
                int(label)
            ] = float(
                probability
            )

        # ----------------------------
        # Human-readable labels
        # ----------------------------

        output["team_a_win"] = output[
            "probabilities"
        ].get(
            1,
            0.0
        )

        output["draw"] = output[
            "probabilities"
        ].get(
            0,
            0.0
        )

        output["team_b_win"] = output[
            "probabilities"
        ].get(
            -1,
            0.0
        )

        output["classes"] = list(
            map(int, self.class_labels)
        )

        output["values"] = list(
            map(float, probabilities)
        )

        output["classes"] = [
            int(x)
            for x in self.class_labels
        ]

        output["values"] = [
            float(x)
            for x in probabilities
        ]

        return output

    #load the world cup 2026 teams
    def load_2026_teams(self):
        """
        Load FIFA World Cup 2026 groups.
        """

        self.groups = {

            "A": ["Czechia", "Mexico", "South Korea", "South Africa"],
            "B": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"],
            "C": ["Brazil", "Haiti", "Morocco", "Scotland"],
            "D": ["Australia", "Paraguay", "Turkiye", "USA"],
            "E": ["Curacao", "Ecuador", "Germany", "Ivory Coast"],
            "F": ["Japan", "Netherlands", "Sweden", "Tunisia"],
            "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
            "H": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
            "I": ["France", "Iraq", "Norway", "Senegal"],
            "J": ["Algeria", "Argentina", "Austria", "Jordan"],
            "K": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
            "L": ["Croatia", "England", "Ghana", "Panama"]

        }

        self.group_tables = {}

        self.group_matches = {}

        for group_name, teams in self.groups.items():

            self.group_tables[group_name] = {}

            self.group_matches[group_name] = []

            for team in teams:

                self.group_tables[group_name][team] = {

                    "played": 0,

                    "wins": 0,

                    "draws": 0,

                    "losses": 0,

                    "gf": 0,

                    "ga": 0,

                    "gd": 0,

                    "points": 0

                }

        self.knockout_bracket = {}
        self.group_match_results = {
            "A": [],
            "B": [],
            "C": [],
            "D": [],
            "E": [],
            "F": [],
            "G": [],
            "H": [],
            "I": [],
            "J": [],
            "K": [],
            "L": [],
        }

        return self.groups

    #simulate group matches
    def simulate_group_match(self, team_a, team_b, group):
        proba = self.predict_match_proba(
            team_a, team_b, stage_weight=self.stage_weights["group"]
        )

        outcome = np.random.choice(proba["classes"], p=proba["values"])

        if outcome == 1:
            scores = [(1, 0), (2, 0), (2, 1), (3, 1), (3, 0)]
        elif outcome == 0:
            scores = [(0, 0), (1, 1), (2, 2)]
        else:
            scores = [(0, 1), (0, 2), (1, 2), (1, 3), (0, 3)]

        goals_a, goals_b = scores[np.random.randint(len(scores))]

        match = {
            "group": group,
            "team_a": team_a,
            "team_b": team_b,
            "goals_a": goals_a,
            "goals_b": goals_b,
            "outcome": int(outcome),
        }

        self._update_group_match(match)

        self.group_match_results[group].append(match)

        return match

    #Helper function for the simulate teams
    def _update_group_match(self, match):
        g = match["group"]
        a = match["team_a"]
        b = match["team_b"]

        goals_a = match["goals_a"]
        goals_b = match["goals_b"]

        A = self.group_tables[g][a]
        B = self.group_tables[g][b]

        # update played
        A["played"] += 1
        B["played"] += 1

        # goals
        A["gf"] += goals_a
        A["ga"] += goals_b

        B["gf"] += goals_b
        B["ga"] += goals_a

        # goal difference
        A["gd"] = A["gf"] - A["ga"]
        B["gd"] = B["gf"] - B["ga"]

        # result handling
        if goals_a > goals_b:
            A["wins"] += 1
            B["losses"] += 1
            A["points"] += 3
        elif goals_a < goals_b:
            B["wins"] += 1
            A["losses"] += 1
            B["points"] += 3
        else:
            A["draws"] += 1
            B["draws"] += 1
            A["points"] += 1
            B["points"] += 1

    #simulate group matches
    def simulate_group(self, group):
        # safety check
        if group not in self.groups:
            raise ValueError(f"Group {group} not found. Run load_2026_teams().")

        teams = self.groups[group]

        if len(teams) != 4:
            raise ValueError(f"Group {group} must have 4 teams.")

        fixtures = list(combinations(teams, 2))

        results = []

        for team_a, team_b in fixtures:
            match = self.simulate_group_match(team_a, team_b, group)
            results.append(match)

        return results

    #build the group table
    def build_group_table(self, group):
        if group not in self.group_tables:
            raise ValueError(f"Group {group} not found.")

        table = self.group_tables[group]

        standings = [
            {"team": team, **stats} for team, stats in table.items()
        ]

        # -------------------------
        # STEP 1: initial sort
        # -------------------------
        standings.sort(
            key=lambda x: (x["points"], x["gd"], x["gf"]), reverse=True
        )

        # -------------------------
        # STEP 2: detect ties
        # -------------------------
        i = 0
        final = []

        while i < len(standings):
            group_block = [standings[i]]
            j = i + 1

            # collect tied teams on points
            while (
                j < len(standings)
                and standings[j]["points"] == standings[i]["points"]
            ):
                group_block.append(standings[j])
                j += 1

            # if tie group > 1 → apply H2H
            if len(group_block) > 1:
                tied_teams = [t["team"] for t in group_block]
                matches = self.group_match_results[group]

                h2h = self._calculate_head_to_head(
                    table, tied_teams, matches
                )

                # rebuild using H2H ranking
                group_block.sort(
                    key=lambda x: (
                        h2h[x["team"]]["points"],
                        h2h[x["team"]]["gd"],
                        h2h[x["team"]]["gf"],
                        x["gd"],
                        x["gf"],
                    ),
                    reverse=True,
                )

            final.extend(group_block)
            i = j

        # -------------------------
        # STEP 3: assign positions
        # -------------------------
        for idx, row in enumerate(final, start=1):
            row["position"] = idx

        return final

    #group stage qualifiers
    def get_group_qualifiers(self):
        top_two = []
        third_place_teams = []

        # loop through all groups
        for group in self.groups.keys():
            table = self.build_group_table(group)

            # ensure sorted standings exist
            if len(table) != 4:
                raise ValueError(f"Group {group} table invalid.")

            # top 2 qualify directly
            top_two.extend(table[:2])

            # third place candidate
            third_place_teams.append(table[2])

        # -----------------------------
        # rank third-place teams globally
        # -----------------------------
        third_place_teams.sort(
            key=lambda x: (x["points"], x["gd"], x["gf"]), reverse=True
        )

        best_third = third_place_teams[:8]

        # -----------------------------
        # final 32 teams
        # -----------------------------
        qualifiers = top_two + best_third

        return {
            "top_two": top_two,
            "third_place": third_place_teams,
            "qualified": qualifiers,
        }

    # inculcate fifa's rules
    def _calculate_head_to_head(self, group_table, tied_teams, group_matches):
        h2h_stats = {
            team: {"points": 0, "gf": 0, "ga": 0, "gd": 0}
            for team in tied_teams
        }

        # filter matches involving tied teams only
        for match in group_matches:
            a = match["team_a"]
            b = match["team_b"]

            if a not in tied_teams or b not in tied_teams:
                continue

            goals_a = match["goals_a"]
            goals_b = match["goals_b"]

            # update goals
            h2h_stats[a]["gf"] += goals_a
            h2h_stats[a]["ga"] += goals_b

            h2h_stats[b]["gf"] += goals_b
            h2h_stats[b]["ga"] += goals_a

            # update points
            if goals_a > goals_b:
                h2h_stats[a]["points"] += 3
            elif goals_a < goals_b:
                h2h_stats[b]["points"] += 3
            else:
                h2h_stats[a]["points"] += 1
                h2h_stats[b]["points"] += 1

        # compute goal difference
        for team in tied_teams:
            h2h_stats[team]["gd"] = (
                h2h_stats[team]["gf"] - h2h_stats[team]["ga"]
            )

        return h2h_stats

    #get group qualifiers
    def get_group_qualifiers(self):
        top_two = []
        third_place = []

        # --------------------------
        # STEP 1: extract from groups
        # --------------------------
        for group in self.groups.keys():
            table = self.build_group_table(group)

            if len(table) != 4:
                raise ValueError(f"Invalid group table for {group}")

            top_two.extend(table[:2])
            third_place.append(table[2])

        # --------------------------
        # STEP 2: rank third place teams
        # --------------------------
        third_place.sort(
            key=lambda x: (x["points"], x["gd"], x["gf"]), reverse=True
        )

        best_8_third = third_place[:8]

        # --------------------------
        # STEP 3: combine
        # --------------------------
        qualified = top_two + best_8_third

        # --------------------------
        # STEP 4: sanity check
        # --------------------------
        if len(qualified) != 32:
            raise ValueError(f"Expected 32 teams, got {len(qualified)}")

        return {
            "top_two": top_two,
            "third_place_all": third_place,
            "best_third": best_8_third,
            "qualified": qualified,
        }

    #Generate random knockout brackets
    def generate_knockout_bracket(self, qualifiers):
        teams = qualifiers["qualified"].copy()

        if len(teams) != 32:
            raise ValueError("Knockout stage requires 32 teams.")

        # --------------------------
        # STEP 1: separate types
        # --------------------------
        top_two = qualifiers["top_two"]
        third = qualifiers["best_third"]

        # --------------------------
        # STEP 2: mild shuffle for realism
        # --------------------------
        random.shuffle(top_two)
        random.shuffle(third)

        # mix while keeping balance
        mixed = top_two + third
        random.shuffle(mixed)

        # --------------------------
        # STEP 3: create matches
        # --------------------------
        matches = []

        for i in range(0, 32, 2):
            match = {
                "team_a": mixed[i]["team"],
                "team_b": mixed[i + 1]["team"],
                "stage": "Round of 32",
            }
            matches.append(match)

        return matches

    #simulate round of 32 matches
    def simulate_round(self, matches, stage_weight=3):
        winners = []
        results = []

        for match in matches:
            team_a = match["team_a"]
            team_b = match["team_b"]

            # --------------------------
            # Predict outcome probabilities
            # --------------------------
            proba = self.predict_match_proba(
                team_a, team_b, stage_weight=stage_weight
            )

            outcome = np.random.choice(proba["classes"], p=proba["values"])

            # --------------------------
            # Determine winner
            # --------------------------
            if outcome == 1:
                winner = team_a
            elif outcome == -1:
                winner = team_b
            else:
                # draw → resolve via probability bias
                winner = np.random.choice([team_a, team_b])

            winners.append(winner)

            results.append(
                {
                    "team_a": team_a,
                    "team_b": team_b,
                    "winner": winner,
                    "stage": match.get("stage", "Knockout"),
                }
            )

        return {"winners": winners, "results": results}

    #Simulate whole tournament
    def simulate_tournament(self, qualifiers):
        # --------------------------
        # Round of 32
        # --------------------------
        bracket = self.generate_knockout_bracket(qualifiers)
        r32 = self.simulate_round(bracket, stage_weight=3)
        teams = r32["winners"]

        # --------------------------
        # Round of 16
        # --------------------------
        r16_matches = self._pair_round(teams)
        r16 = self.simulate_round(r16_matches, stage_weight=4)
        teams = r16["winners"]

        # --------------------------
        # Quarterfinals
        # --------------------------
        qf_matches = self._pair_round(teams)
        qf = self.simulate_round(qf_matches, stage_weight=5)
        teams = qf["winners"]

        # --------------------------
        # Semifinals
        # --------------------------
        sf_matches = self._pair_round(teams)
        sf = self.simulate_round(sf_matches, stage_weight=6)
        teams = sf["winners"]

        # --------------------------
        # Final
        # --------------------------
        final_match = self._pair_round(teams)
        final = self.simulate_round(final_match, stage_weight=7)

        champion = final["winners"][0]

        return {
            "round_of_32": r32,
            "round_of_16": r16,
            "quarterfinals": qf,
            "semifinals": sf,
            "final": final,
            "champion": champion,
        }

    # Pair round
    def _pair_round(self, teams):
        matches = []

        for i in range(0, len(teams), 2):
            matches.append(
                {
                    "team_a": teams[i],
                    "team_b": teams[i + 1],
                    "stage": "Knockout",
                }
            )

        return matches

    # Run many simulations
    def run_many_simulations(self, qualifiers, n_simulations=1000):
        champion_counts = defaultdict(int)

        all_results = []

        for i in range(n_simulations):
            result = self.simulate_tournament(qualifiers)

            champion = result["champion"]

            champion_counts[champion] += 1

            all_results.append(champion)

        # --------------------------
        # convert to probabilities
        # --------------------------
        probabilities = []

        for team, count in champion_counts.items():
            probabilities.append(
                {
                    "team": team,
                    "wins": count,
                    "probability": round(count / n_simulations * 100, 2),
                }
            )

        # sort descending
        probabilities.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "raw_counts": dict(champion_counts),
            "probabilities": probabilities,
            "all_champions": all_results,
        }

    # Plot the results of top ten probabilities to win the tournament
    def plot_probabilities(self, mc_results, top_n=10):
        import matplotlib.pyplot as plt

        top = mc_results["probabilities"][:top_n]

        teams = [x["team"] for x in top]
        values = [x["probability"] for x in top]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(teams, values)

        ax.set_title("World Cup Championship Probability")
        ax.set_ylabel("%")

        plt.xticks(rotation=45)

        return fig

    # Load the saved model
    def load_saved_model(self, model_dir=None):
        import json
        import pickle
        from pathlib import Path

        from catboost import CatBoostClassifier

        if model_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            model_dir = project_root / "models" / "worldcup_predictor"
        else:
            model_dir = Path(model_dir)

        # -----------------------
        # Model
        # -----------------------
        self.model = CatBoostClassifier()
        self.model.load_model(model_dir / "catboost_model.cbm")

        # -----------------------
        # Feature columns
        # -----------------------
        with open(model_dir / "feature_columns.pkl", "rb") as f:
            self.feature_columns = pickle.load(f)

        # -----------------------
        # ELO
        # -----------------------
        with open(model_dir / "current_elo.pkl", "rb") as f:
            self.current_elo = pickle.load(f)

        # -----------------------
        # Team history
        # -----------------------
        with open(model_dir / "team_history.pkl", "rb") as f:
            self.team_history = pickle.load(f)

        # -----------------------
        # H2H history
        # -----------------------
        with open(model_dir / "h2h_history.pkl", "rb") as f:
            self.h2h_history = pickle.load(f)

        # -----------------------
        # Class labels
        # -----------------------
        with open(model_dir / "class_labels.pkl", "rb") as f:
            self.class_labels = pickle.load(f)

        # -----------------------
        # Metadata
        # -----------------------
        with open(model_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        # -----------------------
        # Evaluation Results
        # -----------------------
        evaluation_results_path = model_dir / "evaluation_results.json"
        if evaluation_results_path.exists():
            with open(evaluation_results_path, "r") as f:
                self.evaluation_results = json.load(f)
        else:
            self.evaluation_results = None

        # -----------------------
        # Feature Importance
        # -----------------------
        feature_importance_path = model_dir / "feature_importance.csv"
        if feature_importance_path.exists():
            self.feature_importance_df = pd.read_csv(feature_importance_path)
            self.top_features = self.feature_importance_df.head(10)
        else:
            self.feature_importance_df = None
            self.top_features = None

        print("Saved model loaded successfully.")

__all__ = ["WorldCupPredictor"]
