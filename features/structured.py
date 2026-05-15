# """
# features/structured.py
# ──────────────────────
# Builds the *structured* feature bucket for each node type:
# normalised numerics + one-hot/ordinal categoricals.

# The output is a dict {node_type: List[List[float]]} where
# each inner list has length == STRUCTURED_DIM[node_type].
# """

# from __future__ import annotations

# import logging
# from typing import Any, Dict, List, Optional

# import numpy as np
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# from config.settings import STRUCTURED_DIM

# log = logging.getLogger(__name__)


# # ── ordinal/enum maps ─────────────────────────────────────────────────────────

# EXPERIENCE_LEVEL_MAP = {
#     "junior": 0, "mid": 1, "mid-level": 1, "senior": 2,
#     "lead": 3, "principal": 4, "staff": 4,
# }
# JOB_TYPE_MAP = {
#     "full-time": 0, "part-time": 1, "contract": 2,
#     "internship": 3, "freelance": 4,
# }
# COMPANY_SIZE_MAP = {
#     "startup": 0, "small": 1, "medium": 2,
#     "large": 3, "enterprise": 4,
# }
# DEGREE_LEVEL_MAP = {
#     "bachelor": 0, "licence": 0,
#     "master": 1, "msc": 1,
#     "phd": 2, "doctorate": 2,
# }


# def _safe_float(val, default: float = 0.0) -> float:
#     try:
#         return float(val)
#     except (TypeError, ValueError):
#         return default


# def _ordinal(val: Optional[str], mapping: dict, default: float = 0.0) -> float:
#     if val is None:
#         return default
#     return float(mapping.get(str(val).lower().strip(), default))


# # ── per-type builders ─────────────────────────────────────────────────────────

# def _student_structured(rows: List[Dict[str, Any]]) -> np.ndarray:
#     """[graduation_year_norm, current_year_norm, degree_level_enc]"""
#     raw = np.array([
#         [
#             _safe_float(r.get("graduation_year"), 2024),
#             _safe_float(r.get("current_year_of_study"), 1),
#             _ordinal(r.get("degree_level"), DEGREE_LEVEL_MAP),
#         ]
#         for r in rows
#     ], dtype=np.float32)
#     scaler = MinMaxScaler()
#     raw[:, :2] = scaler.fit_transform(raw[:, :2])
#     return raw


# def _job_structured(rows: List[Dict[str, Any]]) -> np.ndarray:
#     """[salary_norm, remote_bin, experience_level_enc, job_type_enc]"""
#     raw = np.array([
#         [
#             _safe_float(r.get("salary"), 0),
#             1.0 if r.get("remote") else 0.0,
#             _ordinal(r.get("experience_level"), EXPERIENCE_LEVEL_MAP),
#             _ordinal(r.get("job_type"), JOB_TYPE_MAP),
#         ]
#         for r in rows
#     ], dtype=np.float32)
#     # salary column only
#     sal = raw[:, 0:1]
#     if sal.max() > sal.min():
#         raw[:, 0:1] = MinMaxScaler().fit_transform(sal)
#     return raw


# def _company_structured(rows: List[Dict[str, Any]]) -> np.ndarray:
#     """[size_enc]"""
#     raw = np.array([
#         [_ordinal(r.get("size"), COMPANY_SIZE_MAP)]
#         for r in rows
#     ], dtype=np.float32)
#     if raw.max() > 0:
#         raw = raw / raw.max()
#     return raw


# def _skill_structured(rows: List[Dict[str, Any]]) -> np.ndarray:
#     """[layer_norm, type_enc]  — all layer-3, so type_enc always 0"""
#     raw = np.array([
#         [
#             _safe_float(r.get("layer"), 3) / 3.0,
#             0.0,   # always "Specific Skill"
#         ]
#         for r in rows
#     ], dtype=np.float32)
#     return raw


# def _occupation_structured(rows: List[Dict[str, Any]]) -> np.ndarray:
#     """[layer_norm, type_enc]"""
#     return _skill_structured(rows)   # identical logic


# def _zero_structured(rows: List[Dict[str, Any]], dim: int) -> np.ndarray:
#     return np.zeros((len(rows), dim), dtype=np.float32)


# # ── dispatcher ────────────────────────────────────────────────────────────────

# _BUILDERS = {
#     "Student":    _student_structured,
#     "Job":        _job_structured,
#     "Company":    _company_structured,
#     "Skill":      _skill_structured,
#     "Occupation": _occupation_structured,
# }


# def build_structured_features(
#     nodes: Dict[str, List[Dict[str, Any]]],
# ) -> Dict[str, np.ndarray]:
#     """
#     Returns {node_type: np.ndarray [N, structured_dim]} for every type.
#     Types with zero structured features return shape [N, 0].
#     """
#     result: Dict[str, np.ndarray] = {}
#     for ntype, rows in nodes.items():
#         if not rows:
#             result[ntype] = np.zeros((0, STRUCTURED_DIM.get(ntype, 0)), dtype=np.float32)
#             continue

#         sdim = STRUCTURED_DIM.get(ntype, 0)
#         if sdim == 0:
#             result[ntype] = np.zeros((len(rows), 0), dtype=np.float32)
#         elif ntype in _BUILDERS:
#             arr = _BUILDERS[ntype](rows)
#             result[ntype] = arr
#             log.info("Structured features for %s: %s", ntype, arr.shape)
#         else:
#             result[ntype] = np.zeros((len(rows), sdim), dtype=np.float32)
#     return result




# v1


"""
features/structured.py
──────────────────────
Builds the *structured* feature bucket for each node type:
normalised numerics + one-hot/ordinal categoricals.

The output is a dict {node_type: List[List[float]]} where
each inner list has length == STRUCTURED_DIM[node_type].
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from config.settings import STRUCTURED_DIM

log = logging.getLogger(__name__)


# ── ordinal/enum maps ─────────────────────────────────────────────────────────

EXPERIENCE_LEVEL_MAP = {
    "junior": 0, "mid": 1, "mid-level": 1, "senior": 2,
    "lead": 3, "principal": 4, "staff": 4,
}
JOB_TYPE_MAP = {
    "full-time": 0, "part-time": 1, "contract": 2,
    "internship": 3, "freelance": 4,
}
COMPANY_SIZE_MAP = {
    "startup": 0, "small": 1, "medium": 2,
    "large": 3, "enterprise": 4,
}
DEGREE_LEVEL_MAP = {
    "bachelor": 0, "licence": 0,
    "master": 1, "msc": 1,
    "phd": 2, "doctorate": 2,
}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _ordinal(val: Optional[str], mapping: dict, default: float = 0.0) -> float:
    if val is None:
        return default
    return float(mapping.get(str(val).lower().strip(), default))


# ── per-type builders ─────────────────────────────────────────────────────────

def _student_structured(rows: List[Dict[str, Any]]) -> np.ndarray:
    """[graduation_year_norm, current_year_norm, degree_level_enc]"""
    raw = np.array([
        [
            _safe_float(r.get("graduation_year"), 2024),
            _safe_float(r.get("current_year_of_study"), 1),
            _ordinal(r.get("degree_level"), DEGREE_LEVEL_MAP),
        ]
        for r in rows
    ], dtype=np.float32)
    scaler = MinMaxScaler()
    raw[:, :2] = scaler.fit_transform(raw[:, :2])
    return raw


def _job_structured(rows: List[Dict[str, Any]]) -> np.ndarray:
    """[salary_norm, remote_bin, experience_level_enc, job_type_enc]"""
    raw = np.array([
        [
            _safe_float(r.get("salary"), 0),
            1.0 if r.get("remote") else 0.0,
            _ordinal(r.get("experience_level"), EXPERIENCE_LEVEL_MAP),
            _ordinal(r.get("job_type"), JOB_TYPE_MAP),
        ]
        for r in rows
    ], dtype=np.float32)
    # salary column only
    sal = raw[:, 0:1]
    if sal.max() > sal.min():
        raw[:, 0:1] = MinMaxScaler().fit_transform(sal)
    return raw


def _company_structured(rows: List[Dict[str, Any]]) -> np.ndarray:
    """[size_enc]"""
    raw = np.array([
        [_ordinal(r.get("size"), COMPANY_SIZE_MAP)]
        for r in rows
    ], dtype=np.float32)
    if raw.max() > 0:
        raw = raw / raw.max()
    return raw


def _skill_structured(rows: List[Dict[str, Any]]) -> np.ndarray:
    """[layer_norm, type_enc]  — layer value read from row, so works for L1/L2/L3"""
    raw = np.array([
        [
            _safe_float(r.get("layer"), 3) / 3.0,
            0.0,   # type_enc — kept as constant for now
        ]
        for r in rows
    ], dtype=np.float32)
    return raw


def _occupation_structured(rows: List[Dict[str, Any]]) -> np.ndarray:
    """[layer_norm, type_enc] — identical logic to _skill_structured"""
    return _skill_structured(rows)


def _zero_structured(rows: List[Dict[str, Any]], dim: int) -> np.ndarray:
    return np.zeros((len(rows), dim), dtype=np.float32)


# ── dispatcher ────────────────────────────────────────────────────────────────

_BUILDERS = {
    "Student":      _student_structured,
    "Job":          _job_structured,
    # "Company":      _company_structured,
    "Skill_L1":     _skill_structured,
    "Skill_L2":     _skill_structured,
    "Skill_L3":     _skill_structured,
    "Occupation_L1":_occupation_structured,
    "Occupation_L2":_occupation_structured,
    "Occupation_L3":_occupation_structured,
}


def build_structured_features(
    nodes: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, np.ndarray]:
    """
    Returns {node_type: np.ndarray [N, structured_dim]} for every type.
    Types with zero structured features return shape [N, 0].
    """
    result: Dict[str, np.ndarray] = {}
    for ntype, rows in nodes.items():
        if not rows:
            result[ntype] = np.zeros((0, STRUCTURED_DIM.get(ntype, 0)), dtype=np.float32)
            continue

        sdim = STRUCTURED_DIM.get(ntype, 0)
        if sdim == 0:
            result[ntype] = np.zeros((len(rows), 0), dtype=np.float32)
        elif ntype in _BUILDERS:
            arr = _BUILDERS[ntype](rows)
            result[ntype] = arr
            log.info("Structured features for %s: %s", ntype, arr.shape)
        else:
            result[ntype] = np.zeros((len(rows), sdim), dtype=np.float32)
    return result