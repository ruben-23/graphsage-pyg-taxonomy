# """
# features/semantic.py
# ─────────────────────
# Constructs the semantic (text) bucket for each node type.

# Rules
# -----
# * Skill / Occupation  : uses the full taxonomy path
#   e.g. "Hard Skills -> Programming & Scripting -> Python"
# * All other types     : concatenate relevant text fields into a sentence.
# * Missing fields are silently omitted.
# """

# from __future__ import annotations

# import logging
# from typing import Any, Dict, List, Optional

# log = logging.getLogger(__name__)


# def _join(*parts: Optional[str], sep: str = ". ") -> str:
#     """Join non-empty, non-None parts."""
#     return sep.join(p.strip() for p in parts if p and str(p).strip())


# # ── per-type text builders ────────────────────────────────────────────────────

# def _student_text(r: dict) -> str:
#     name   = r.get("name", "")
#     major  = r.get("major", "")
#     degree = r.get("degree_level", "")
#     return _join(
#         f"Student named {name}" if name else "Student",
#         f"studying {major}" if major else None,
#         f"{degree} degree" if degree else None,
#     )


# def _job_text(r: dict) -> str:
#     title  = r.get("title", "")
#     desc   = r.get("description", "")
#     loc    = r.get("location", "")
#     exp    = r.get("experience_level", "")
#     return _join(
#         f"Job title: {title}" if title else "Job posting",
#         f"Location: {loc}" if loc else None,
#         f"Experience level: {exp}" if exp else None,
#         desc if desc else None,
#     )


# def _company_text(r: dict) -> str:
#     name     = r.get("name", "")
#     industry = r.get("industry", "")
#     loc      = r.get("location", "")
#     size     = r.get("size", "")
#     return _join(
#         f"Company: {name}" if name else "Company",
#         f"Industry: {industry}" if industry else None,
#         f"Located in {loc}" if loc else None,
#         f"Size: {size}" if size else None,
#     )


# def _skill_text(r: dict) -> str:
#     """Build taxonomy path: Category -> Group -> Skill name."""
#     name     = r.get("name", "")
#     group    = r.get("group_name", "")
#     category = r.get("category_name", "")

#     if category and group:
#         return f"{category} -> {group} -> {name}"
#     elif group:
#         return f"{group} -> {name}"
#     else:
#         return name or "Skill"
#     # base = f"Skill: {name}."
#     # if category and group:
#     #     # return f"{category} -> {group} -> {name}"
#     #     return f"{base} This belongs to the {group} group under {category}."
#     # elif group:
#     #     # return f"{group} -> {name}"
#     #     return f"{base} This belongs to the {group} group."
#     # return name or "Skill"


# def _occupation_text(r: dict) -> str:
#     """Build taxonomy path: Grandparent -> Parent -> Occupation name."""
#     name        = r.get("name", "")
#     parent      = r.get("parent_name") or r.get("parent", "")
#     grandparent = r.get("grandparent_name", "")

#     if grandparent and parent:
#         return f"{grandparent} -> {parent} -> {name}"
#     elif parent:
#         return f"{parent} -> {name}"
#     else:
#         return name or "Occupation"

#     # base = f"Occupation: {name}."
#     # if grandparent and parent:
#     #     # return f"{grandparent} -> {parent} -> {name}"
#     #     return f"{base} This is a specialization of {parent}, part of the {grandparent} job family."
#     # elif parent:
#     #     # return f"{parent} -> {name}"
#     #     return f"{base} This is a specialization of {parent}."
#     # return name or "Occupation"


# def _project_text(r: dict) -> str:
#     title = r.get("title", "")
#     desc  = r.get("description", "")
#     return _join(
#         f"Project: {title}" if title else "Project",
#         desc if desc else None,
#     )


# def _course_text(r: dict) -> str:
#     title    = r.get("title", "")
#     desc     = r.get("description", "")
#     provider = r.get("provider", "")
#     return _join(
#         f"Course: {title}" if title else "Course",
#         f"Provider: {provider}" if provider else None,
#         desc if desc else None,
#     )


# def _diploma_text(r: dict) -> str:
#     title  = r.get("title", "")
#     desc   = r.get("description", "")
#     issuer = r.get("issuer", "")
#     return _join(
#         f"Diploma: {title}" if title else "Diploma",
#         f"Issued by {issuer}" if issuer else None,
#         desc if desc else None,
#     )


# # ── dispatcher ────────────────────────────────────────────────────────────────

# _TEXT_BUILDERS = {
#     "Student":    _student_text,
#     "Job":        _job_text,
#     "Company":    _company_text,
#     "Skill":      _skill_text,
#     "Occupation": _occupation_text,
#     "Project":    _project_text,
#     "Course":     _course_text,
#     "Diploma":    _diploma_text,
# }


# def build_semantic_texts(
#     nodes: Dict[str, List[Dict[str, Any]]],
# ) -> Dict[str, List[str]]:
#     """
#     Returns {node_type: [text_string, ...]} — one string per node,
#     ready to be embedded.
#     """
#     result: Dict[str, List[str]] = {}
#     for ntype, rows in nodes.items():
#         builder = _TEXT_BUILDERS.get(ntype)
#         if builder is None:
#             result[ntype] = [ntype] * len(rows)
#             continue
#         texts = [builder(r) for r in rows]
#         result[ntype] = texts
#         log.info(
#             "Built %d semantic texts for %s. Sample: %s",
#             len(texts), ntype,
#             texts[0][:80] if texts else "(empty)",
#         )
#     return result



# v1

"""
features/semantic.py
─────────────────────
Constructs the semantic (text) bucket for each node type.

Rules
-----
* Skill / Occupation  : uses the full taxonomy path
  e.g. "Hard Skills -> Programming & Scripting -> Python"
* All other types     : concatenate relevant text fields into a sentence.
* Missing fields are silently omitted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def _join(*parts: Optional[str], sep: str = ". ") -> str:
    """Join non-empty, non-None parts."""
    return sep.join(p.strip() for p in parts if p and str(p).strip())


# ── per-type text builders ────────────────────────────────────────────────────

def _student_text(r: dict) -> str:
    name   = r.get("name", "")
    major  = r.get("major", "")
    degree = r.get("degree_level", "")
    return _join(
        f"Student named {name}" if name else "Student",
        f"studying {major}" if major else None,
        f"{degree} degree" if degree else None,
    )


def _job_text(r: dict) -> str:
    title  = r.get("title", "")
    desc   = r.get("description", "")
    loc    = r.get("location", "")
    exp    = r.get("experience_level", "")
    return _join(
        f"Job title: {title}" if title else "Job posting",
        f"Location: {loc}" if loc else None,
        f"Experience level: {exp}" if exp else None,
        desc if desc else None,
    )


def _company_text(r: dict) -> str:
    name     = r.get("name", "")
    industry = r.get("industry", "")
    loc      = r.get("location", "")
    size     = r.get("size", "")
    return _join(
        f"Company: {name}" if name else "Company",
        f"Industry: {industry}" if industry else None,
        f"Located in {loc}" if loc else None,
        f"Size: {size}" if size else None,
    )


def _skill_l1_text(r: dict) -> str:
    """Top-level skill category — just the name."""
    return r.get("name") or "Skill Category"


def _skill_l2_text(r: dict) -> str:
    """Skill group — Category -> Group."""
    name     = r.get("name", "")
    category = r.get("category_name", "")
    if category:
        return f"{category} -> {name}"
    return name or "Skill Group"


def _skill_text(r: dict) -> str:
    """Build taxonomy path: Category -> Group -> Skill name."""
    name     = r.get("name", "")
    group    = r.get("group_name", "")
    category = r.get("category_name", "")

    # if category and group:
    #     return f"{category} -> {group} -> {name}"
    # elif group:
    #     return f"{group} -> {name}"
    # else:
    #     return name or "Skill"
    return f"Skill: {name}. Group: {group}" if group else name


def _occupation_l1_text(r: dict) -> str:
    """Top-level occupation family — just the name."""
    return r.get("name") or "Occupation Family"


def _occupation_l2_text(r: dict) -> str:
    """Occupation group — Family -> Group."""
    name        = r.get("name", "")
    grandparent = r.get("grandparent_name", "")
    if grandparent:
        return f"{grandparent} -> {name}"
    return name or "Occupation Group"


def _occupation_text(r: dict) -> str:
    """Build taxonomy path: Grandparent -> Parent -> Occupation name."""
    name        = r.get("name", "")
    parent      = r.get("parent_name") or r.get("parent", "")
    grandparent = r.get("grandparent_name", "")

    if grandparent and parent:
        return f"{grandparent} -> {parent} -> {name}"
    elif parent:
        return f"{parent} -> {name}"
    else:
        return name or "Occupation"


def _project_text(r: dict) -> str:
    title = r.get("title", "")
    desc  = r.get("description", "")
    return _join(
        f"Project: {title}" if title else "Project",
        desc if desc else None,
    )


def _course_text(r: dict) -> str:
    title    = r.get("title", "")
    desc     = r.get("description", "")
    provider = r.get("provider", "")
    return _join(
        f"Course: {title}" if title else "Course",
        f"Provider: {provider}" if provider else None,
        desc if desc else None,
    )


def _diploma_text(r: dict) -> str:
    title  = r.get("title", "")
    desc   = r.get("description", "")
    issuer = r.get("issuer", "")
    return _join(
        f"Diploma: {title}" if title else "Diploma",
        f"Issued by {issuer}" if issuer else None,
        desc if desc else None,
    )


# ── dispatcher ────────────────────────────────────────────────────────────────

_TEXT_BUILDERS = {
    "Student":      _student_text,
    "Job":          _job_text,
    "Company":      _company_text,
    "Skill_L1":     _skill_l1_text,
    "Skill_L2":     _skill_l2_text,
    "Skill_L3":     _skill_text,
    "Occupation_L1":_occupation_l1_text,
    "Occupation_L2":_occupation_l2_text,
    "Occupation_L3":_occupation_text,
    "Project":      _project_text,
    "Course":       _course_text,
    "Diploma":      _diploma_text,
}


def build_semantic_texts(
    nodes: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[str]]:
    """
    Returns {node_type: [text_string, ...]} — one string per node,
    ready to be embedded.
    """
    result: Dict[str, List[str]] = {}
    for ntype, rows in nodes.items():
        builder = _TEXT_BUILDERS.get(ntype)
        if builder is None:
            result[ntype] = [ntype] * len(rows)
            continue
        texts = [builder(r) for r in rows]
        result[ntype] = texts
        log.info(
            "Built %d semantic texts for %s. Sample: %s",
            len(texts), ntype,
            texts[0][:80] if texts else "(empty)",
        )
    return result