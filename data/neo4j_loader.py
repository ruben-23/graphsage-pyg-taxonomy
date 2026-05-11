
# original version

# """
# data/neo4j_loader.py
# ────────────────────
# Loads all node types and relationships from Neo4j and returns raw dicts.
# Only layer-3 Skill and Occupation nodes are kept; taxonomy path is
# reconstructed from the graph so it can be embedded as part of the text.
# """

# from __future__ import annotations

# import logging
# from typing import Any, Dict, List, Tuple

# from neo4j import GraphDatabase

# from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# log = logging.getLogger(__name__)


# # ── helpers ──────────────────────────────────────────────────────────────────

# def _run(session, cypher: str, **params) -> List[dict]:
#     result = session.run(cypher, **params)
#     return [dict(r) for r in result]


# # ── node queries ─────────────────────────────────────────────────────────────

# _NODE_QUERIES: Dict[str, str] = {
#     "Student": """
#         MATCH (s:Student)
#         RETURN s.student_id          AS student_id,
#                s.name                AS name,
#                s.major               AS major,
#                s.graduation_year     AS graduation_year,
#                s.current_year_of_study AS current_year_of_study,
#                s.degree_level        AS degree_level
#     """,
#     "Job": """
#         MATCH (j:Job)
#         RETURN j.job_id            AS job_id,
#                j.title             AS title,
#                j.description       AS description,
#                j.experience_level  AS experience_level,
#                j.job_type          AS job_type,
#                j.salary            AS salary,
#                j.currency          AS currency,
#                j.remote            AS remote,
#                j.location          AS location
#     """,
#     "Company": """
#         MATCH (c:Company)
#         RETURN c.company_id  AS company_id,
#                c.name        AS name,
#                c.industry    AS industry,
#                c.location    AS location,
#                c.size        AS size
#     """,
#     # Only layer-3 skills (Specific Skill)
#     "Skill": """
#         MATCH (sk:Skill {layer: 3})
#         OPTIONAL MATCH (sk)-[:SUBCLASS_OF]->(sg:Skill {layer: 2})-[:SUBCLASS_OF]->(sc:Skill {layer: 1})
#         RETURN sk.skill_id  AS skill_id,
#                sk.name      AS name,
#                sk.layer     AS layer,
#                sk.type      AS type,
#                sk.parent    AS parent,
#                sg.name      AS group_name,
#                sc.name      AS category_name
#     """,
#     # Only layer-3 occupations (Specialized Role)
#     "Occupation": """
#         MATCH (o:Occupation {layer: 3})
#         // The original query with two independent OPTIONAL MATCHes could create
#         // a cartesian product, duplicating rows. This chained version is safer.
#         // It also changes `*` to `*1..` to require a path of at least one hop.
#         OPTIONAL MATCH (o)-[:SUBCLASS_OF*1..]->(p:Occupation {layer: 2})
#         OPTIONAL MATCH (p)-[:SUBCLASS_OF*1..]->(gp:Occupation {layer: 1})
#         RETURN o.occupation_id  AS occupation_id,
#                o.name           AS name,
#                o.layer          AS layer,
#                o.type           AS type,
#                o.parent         AS parent,
#                p.name           AS parent_name,
#                gp.name          AS grandparent_name
#     """,
#     "Project": """
#         MATCH (p:Project)
#         RETURN p.project_id   AS project_id,
#                p.title        AS title,
#                p.description  AS description
#     """,
#     "Course": """
#         MATCH (c:Course)
#         RETURN c.course_id   AS course_id,
#                c.title       AS title,
#                c.description AS description,
#                c.provider    AS provider
#     """,
#     "Diploma": """
#         MATCH (d:Diploma)
#         RETURN d.diploma_id  AS diploma_id,
#                d.title       AS title,
#                d.description AS description,
#                d.issuer      AS issuer
#     """,
# }

# # ── relationship queries ──────────────────────────────────────────────────────

# _REL_QUERIES: Dict[str, Tuple[str, str, str]] = {
#     # (src_type, dst_type, cypher_returning src_id, dst_id + optional props)
#     "KNOWS": (
#         "Student", "Skill",
#         """
#         MATCH (s:Student)-[r:KNOWS]->(sk:Skill {layer: 3})
#         RETURN s.student_id AS src, sk.skill_id AS dst,
#                r.proficiency_level AS level, r.years_0f_experience AS years_of_experience
#         """,
#     ),
#     "REQUIRES": (
#         "Job", "Skill",
#         """
#         MATCH (j:Job)-[r:REQUIRES]->(sk:Skill {layer: 3})
#         RETURN j.job_id AS src, sk.skill_id AS dst,
#                r.min_proficiency AS level, r.importance AS importance
#         """,
#     ),
#     "POSTS": (
#         "Company", "Job",
#         """
#         MATCH (c:Company)-[:POSTS]->(j:Job)
#         RETURN c.company_id AS src, j.job_id AS dst
#         """,
#     ),
#     "CREATED": (
#         "Student", "Project",
#         """
#         MATCH (s:Student)-[:CREATED]->(p:Project)
#         RETURN s.student_id AS src, p.project_id AS dst
#         """,
#     ),
#     "BUILT_WITH": (
#         "Project", "Skill",
#         """
#         MATCH (p:Project)-[:BUILT_WITH]->(sk:Skill {layer: 3})
#         RETURN p.project_id AS src, sk.skill_id AS dst
#         """,
#     ),
#     "COMPLETED": (
#         "Student", "Course",
#         """
#         MATCH (s:Student)-[:COMPLETED]->(c:Course)
#         RETURN s.student_id AS src, c.course_id AS dst
#         """,
#     ),
#     "EARNED": (
#         "Student", "Diploma",
#         """
#         MATCH (s:Student)-[:EARNED]->(d:Diploma)
#         RETURN s.student_id AS src, d.diploma_id AS dst
#         """,
#     ),
#     # "COVERS": (
#     #     "Course", "Skill",
#     #     """
#     #     MATCH (c:Course)-[:COVERS]->(sk:Skill {layer: 3})
#     #     RETURN c.course_id AS src, sk.skill_id AS dst
#     #     """,
#     # ),
#     "CERTIFIES": (
#         "Diploma", "Skill",
#         """
#         MATCH (d:Diploma)-[:CERTIFIES]->(sk:Skill {layer: 3})
#         RETURN d.diploma_id AS src, sk.skill_id AS dst
#         """,
#     ),
# }


# class Neo4jLoader:
#     def __init__(
#         self,
#         uri: str = NEO4J_URI,
#         user: str = NEO4J_USER,
#         password: str = NEO4J_PASSWORD,
#     ):
#         self._driver = GraphDatabase.driver(uri, auth=(user, password))
#         log.info("Connected to Neo4j at %s", uri)

#     def close(self):
#         self._driver.close()

#     # ── public API ────────────────────────────────────────────────────────────

#     def load_nodes(self) -> Dict[str, List[Dict[str, Any]]]:
#         """Return {node_type: [row_dict, ...]} for every node type."""
#         nodes: Dict[str, List[Dict[str, Any]]] = {}
#         with self._driver.session() as session:
#             for ntype, cypher in _NODE_QUERIES.items():
#                 rows = _run(session, cypher)
#                 nodes[ntype] = rows
#                 log.info("Loaded %d %s nodes", len(rows), ntype)
#         return nodes

#     def load_edges(self) -> Dict[str, List[Dict[str, Any]]]:
#         """
#         Return {rel_type: [{'src': id, 'dst': id, ...props}, ...]} for every
#         relationship type.  Also includes src_type / dst_type meta-fields.
#         """
#         edges: Dict[str, List[Dict[str, Any]]] = {}
#         with self._driver.session() as session:
#             for rel, (src_t, dst_t, cypher) in _REL_QUERIES.items():
#                 rows = _run(session, cypher)
#                 for r in rows:
#                     r["src_type"] = src_t
#                     r["dst_type"] = dst_t
#                 edges[rel] = rows
#                 log.info("Loaded %d %s edges", len(rows), rel)
#         return edges

#     def write_embeddings(
#         self,
#         node_type: str,
#         id_field: str,
#         embeddings: Dict[str, List[float]],
#         property_name: str = "graphsage_embedding",
#     ) -> None:
#         """Batch-write embeddings back to Neo4j."""
#         label = node_type
#         cypher = (
#             f"UNWIND $rows AS row "
#             f"MATCH (n:{label} {{{id_field}: row.id}}) "
#             f"SET n.{property_name} = row.embedding"
#         )
#         rows = [{"id": nid, "embedding": emb} for nid, emb in embeddings.items()]
#         with self._driver.session() as session:
#             # batch in chunks of 500
#             for i in range(0, len(rows), 500):
#                 session.run(cypher, rows=rows[i : i + 500])
#         log.info("Wrote %d %s embeddings -> Neo4j", len(rows), node_type)


# v1

"""
data/neo4j_loader.py
────────────────────
Loads all node types and relationships from Neo4j and returns raw dicts.
Only layer-3 Skill and Occupation nodes are kept; taxonomy path is
reconstructed from the graph so it can be embedded as part of the text.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _run(session, cypher: str, **params) -> List[dict]:
    result = session.run(cypher, **params)
    return [dict(r) for r in result]


# ── node queries ─────────────────────────────────────────────────────────────

_NODE_QUERIES: Dict[str, str] = {
    "Student": """
        MATCH (s:Student)
        RETURN trim(s.student_id)    AS student_id,
               s.name                AS name,
               s.major               AS major,
               s.graduation_year     AS graduation_year,
               s.current_year_of_study AS current_year_of_study,
               s.degree_level        AS degree_level
    """,
    "Job": """
        MATCH (j:Job)
        RETURN trim(j.job_id)      AS job_id,
               j.title             AS title,
               j.clean_description       AS description,
               j.experience_level  AS experience_level,
               j.job_type          AS job_type,
               j.salary            AS salary,
               j.currency          AS currency,
               j.remote            AS remote,
               j.location          AS location
    """,
    "Company": """
        MATCH (c:Company)
        RETURN trim(c.company_id)  AS company_id,
               c.name        AS name,
               c.industry    AS industry,
               c.location    AS location,
               c.size        AS size
    """,
    # ── Skills (one query per layer) ──────────────────────────────────────────
    "Skill_L1": """
        MATCH (sc:Skill {layer: 1})
        RETURN trim(sc.skill_id) AS skill_id,
               sc.name     AS name,
               sc.layer    AS layer,
               sc.type     AS type,
               null        AS parent,
               null        AS group_name,
               null        AS category_name
    """,
    "Skill_L2": """
        MATCH (sg:Skill {layer: 2})
        OPTIONAL MATCH (sg)-[:SUBCLASS_OF]->(sc:Skill {layer: 1})
        RETURN trim(sg.skill_id) AS skill_id,
               sg.name     AS name,
               sg.layer    AS layer,
               sg.type     AS type,
               sg.parent   AS parent,
               null        AS group_name,
               sc.name     AS category_name
    """,
    "Skill_L3": """
        MATCH (sk:Skill {layer: 3})
        OPTIONAL MATCH (sk)-[:SUBCLASS_OF]->(sg:Skill {layer: 2})-[:SUBCLASS_OF]->(sc:Skill {layer: 1})
        RETURN trim(sk.skill_id) AS skill_id,
               sk.name     AS name,
               sk.layer    AS layer,
               sk.type     AS type,
               sk.parent   AS parent,
               sg.name     AS group_name,
               sc.name     AS category_name
    """,
    # ── Occupations (one query per layer) ─────────────────────────────────────
    "Occupation_L1": """
        MATCH (o:Occupation {layer: 1})
        RETURN trim(o.occupation_id) AS occupation_id,
               o.name          AS name,
               o.layer         AS layer,
               o.type          AS type,
               null            AS parent,
               null            AS parent_name,
               null            AS grandparent_name
    """,
    "Occupation_L2": """
        MATCH (o:Occupation {layer: 2})
        OPTIONAL MATCH (o)-[:SUBCLASS_OF*1..]->(gp:Occupation {layer: 1})
        RETURN trim(o.occupation_id) AS occupation_id,
               o.name          AS name,
               o.layer         AS layer,
               o.type          AS type,
               o.parent        AS parent,
               null            AS parent_name,
               gp.name         AS grandparent_name
    """,
    "Occupation_L3": """
        MATCH (o:Occupation {layer: 3})
        OPTIONAL MATCH (o)-[:SUBCLASS_OF*1..]->(p:Occupation {layer: 2})
        OPTIONAL MATCH (p)-[:SUBCLASS_OF*1..]->(gp:Occupation {layer: 1})
        RETURN trim(o.occupation_id) AS occupation_id,
               o.name          AS name,
               o.layer         AS layer,
               o.type          AS type,
               o.parent        AS parent,
               p.name          AS parent_name,
               gp.name         AS grandparent_name
    """,
    "Project": """
        MATCH (p:Project)
        RETURN trim(p.project_id)   AS project_id,
               p.title        AS title,
               p.description  AS description
    """,
    "Course": """
        MATCH (c:Course)
        RETURN trim(c.course_id)   AS course_id,
               c.title       AS title,
               c.description AS description,
               c.provider    AS provider
    """,
    "Diploma": """
        MATCH (d:Diploma)
        RETURN trim(d.diploma_id)  AS diploma_id,
               d.title       AS title,
               d.description AS description,
               d.issuer      AS issuer
    """,
}

# ── relationship queries ──────────────────────────────────────────────────────

_REL_QUERIES: Dict[str, Tuple[str, str, str]] = {
    # (src_type, dst_type, cypher_returning src_id, dst_id + optional props)
    "KNOWS": (
        "Student", "Skill_L3",
        """
        MATCH (s:Student)-[r:KNOWS]->(sk:Skill {layer: 3})
        RETURN s.student_id AS src, sk.skill_id AS dst,
               r.proficiency_level AS level, r.years_of_experience AS years_of_experience
        """,
    ),
    "REQUIRES": (
        "Job", "Skill_L3",
        """
        MATCH (j:Job)-[r:REQUIRES]->(sk:Skill {layer: 3})
        RETURN j.job_id AS src, sk.skill_id AS dst,
               r.min_proficiency AS level, r.importance AS importance
        """,
    ),
    "POSTS": (
        "Company", "Job",
        """
        MATCH (c:Company)-[:POSTS]->(j:Job)
        RETURN c.company_id AS src, j.job_id AS dst
        """,
    ),
    "CREATED": (
        "Student", "Project",
        """
        MATCH (s:Student)-[:CREATED]->(p:Project)
        RETURN s.student_id AS src, p.project_id AS dst
        """,
    ),
    "BUILT_WITH": (
        "Project", "Skill_L3",
        """
        MATCH (p:Project)-[:BUILT_WITH]->(sk:Skill {layer: 3})
        RETURN p.project_id AS src, sk.skill_id AS dst
        """,
    ),
    "COMPLETED": (
        "Student", "Course",
        """
        MATCH (s:Student)-[:COMPLETED]->(c:Course)
        RETURN s.student_id AS src, c.course_id AS dst
        """,
    ),
    "EARNED": (
        "Student", "Diploma",
        """
        MATCH (s:Student)-[:EARNED]->(d:Diploma)
        RETURN s.student_id AS src, d.diploma_id AS dst
        """,
    ),
    # "COVERS": (
    #     "Course", "Skill_L3",
    #     """
    #     MATCH (c:Course)-[:COVERS]->(sk:Skill {layer: 3})
    #     RETURN c.course_id AS src, sk.skill_id AS dst
    #     """,
    # ),
    "CERTIFIES": (
        "Diploma", "Skill_L3",
        """
        MATCH (d:Diploma)-[:CERTIFIES]->(sk:Skill {layer: 3})
        RETURN d.diploma_id AS src, sk.skill_id AS dst
        """,
    ),
    # ── Skill hierarchy ───────────────────────────────────────────────────────
    "SKILL_SUBCLASS_L3_L2": (
        "Skill_L3", "Skill_L2",
        """
        MATCH (sk:Skill {layer: 3})-[:SUBCLASS_OF]->(sg:Skill {layer: 2})
        RETURN sk.skill_id AS src, sg.skill_id AS dst
        """,
    ),
    "SKILL_SUBCLASS_L2_L1": (
        "Skill_L2", "Skill_L1",
        """
        MATCH (sg:Skill {layer: 2})-[:SUBCLASS_OF]->(sc:Skill {layer: 1})
        RETURN sg.skill_id AS src, sc.skill_id AS dst
        """,
    ),
    # ── Occupation hierarchy ──────────────────────────────────────────────────
    "OCC_SUBCLASS_L3_L2": (
        "Occupation_L3", "Occupation_L2",
        """
        MATCH (o3:Occupation {layer: 3})-[:SUBCLASS_OF]->(o2:Occupation {layer: 2})
        RETURN o3.occupation_id AS src, o2.occupation_id AS dst
        """,
    ),
    "OCC_SUBCLASS_L2_L1": (
        "Occupation_L2", "Occupation_L1",
        """
        MATCH (o2:Occupation {layer: 2})-[:SUBCLASS_OF]->(o1:Occupation {layer: 1})
        RETURN o2.occupation_id AS src, o1.occupation_id AS dst
        """,
    ),
}


class Neo4jLoader:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        log.info("Connected to Neo4j at %s", uri)

    def close(self):
        self._driver.close()

    # ── public API ────────────────────────────────────────────────────────────

    def load_nodes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return {node_type: [row_dict, ...]} for every node type."""
        nodes: Dict[str, List[Dict[str, Any]]] = {}
        with self._driver.session() as session:
            for ntype, cypher in _NODE_QUERIES.items():
                rows = _run(session, cypher)
                nodes[ntype] = rows
                log.info("Loaded %d %s nodes", len(rows), ntype)
        return nodes

    def load_edges(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return {rel_type: [{'src': id, 'dst': id, ...props}, ...]} for every
        relationship type.  Also includes src_type / dst_type meta-fields.
        """
        edges: Dict[str, List[Dict[str, Any]]] = {}
        with self._driver.session() as session:
            for rel, (src_t, dst_t, cypher) in _REL_QUERIES.items():
                rows = _run(session, cypher)
                for r in rows:
                    r["src_type"] = src_t
                    r["dst_type"] = dst_t
                edges[rel] = rows
                log.info("Loaded %d %s edges", len(rows), rel)
        return edges

    def write_embeddings(
        self,
        node_type: str,
        id_field: str,
        embeddings: Dict[str, List[float]],
        property_name: str = "graphsage_embedding",
    ) -> None:
        """Batch-write embeddings back to Neo4j."""
        label = node_type
        cypher = (
            f"UNWIND $rows AS row "
            f"MATCH (n:{label}) WHERE trim(n.{id_field}) = row.id "
            f"SET n.{property_name} = row.embedding"
        )
        rows = [{"id": nid, "embedding": emb} for nid, emb in embeddings.items()]
        with self._driver.session() as session:
            # batch in chunks of 500
            for i in range(0, len(rows), 500):
                session.run(cypher, rows=rows[i : i + 500])
        log.info("Wrote %d %s embeddings -> Neo4j", len(rows), node_type)