class FeatureStoreError(Exception):
    pass


class NotFoundError(FeatureStoreError):
    pass


class MissingFeatureError(FeatureStoreError):
    pass


class SchemaMismatchError(FeatureStoreError):
    pass


REQUIRED_SCHEMA = {
    "schema_version": "1.0",
    "spatial": {
        "policy_map": {"shape": [19, 19], "dtype": "float32"},
        "ownership_map": {"shape": [19, 19], "dtype": "float32", "optional": True},
        "legal_mask": {"shape": [19, 19], "dtype": "bool"},
    },
    "scalars": [
        "position_id",
        "game_id",
        "turn_number",
        "split",
        "player_to_move",
        "human_move",
        "human_move_index",
        "root_winrate",
        "root_score_lead",
        "policy_entropy",
        "human_move_policy",
        "human_move_rank_policy",
        "profile_id",
        "schema_version",
    ],
}

