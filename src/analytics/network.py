"""Transaction network structure.

This module exists to answer one question honestly: **does this dataset contain
a fraud ring?** The competition brief assumes it does. The forensic audit found no
statistically or structurally supported evidence for the tested ring pattern:

  * the graph is strictly BIPARTITE (users -> merchants). No ID appears on both
    sides, so there are no user-to-user transfers and no directed money path
    of the A -> B -> C -> A shape the brief describes.
  * every transaction is a UNIQUE (user, merchant) pair. Not one pair repeats.
  * no two users share two or more merchants, so the bipartite graph contains
    ZERO 4-cycles — the shortest cycle a bipartite graph can have. The graph is
    a forest.
  * the largest connected component holds a tiny fraction of the nodes, so
    community detection has nothing to partition.

Rather than manufacture a ring, we compute these numbers and publish them as a
falsified hypothesis. Everything here is derived from the Stage 2 star schema.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import networkx as nx
import pandas as pd

# Guard against the combinatorial blowup of enumerating pairs under a very
# high-degree merchant. Well above the observed maximum degree of 10.
MAX_DEGREE_FOR_PAIR_ENUMERATION = 200


def build_graph(tx: pd.DataFrame) -> nx.Graph:
    """Undirected bipartite graph: user nodes <-> merchant nodes."""
    g = nx.Graph()
    users = tx["user_id_normalized"].dropna()
    merchants = tx["merchant_id_normalized"].dropna()
    g.add_nodes_from(users.unique(), kind="USER")
    g.add_nodes_from(merchants.unique(), kind="MERCHANT")
    pairs = tx.dropna(subset=["user_id_normalized", "merchant_id_normalized"])
    weights = pairs.groupby(["user_id_normalized", "merchant_id_normalized"]).size()
    g.add_edges_from((u, m, {"transactions": int(w)}) for (u, m), w in weights.items())
    return g


def count_four_cycles(tx: pd.DataFrame) -> dict:
    """Count user pairs sharing 2+ merchants — the shortest possible cycle here.

    A bipartite graph has no odd cycles at all, so a 4-cycle
    (u1 - m1 - u2 - m2 - u1) is the minimal closed loop. Zero of them means the
    graph is acyclic and no 'circular money movement' pattern can exist.
    """
    pairs = tx.dropna(subset=["user_id_normalized", "merchant_id_normalized"])
    by_merchant = pairs.groupby("merchant_id_normalized")["user_id_normalized"].apply(
        lambda s: sorted(set(s))
    )
    shared = Counter()
    skipped = 0
    for users in by_merchant:
        if len(users) < 2:
            continue
        if len(users) > MAX_DEGREE_FOR_PAIR_ENUMERATION:
            skipped += 1
            continue
        for a, b in combinations(users, 2):
            shared[(a, b)] += 1
    counts = Counter(shared.values())
    return {
        "user_pairs_sharing_a_merchant": len(shared),
        "user_pairs_sharing_2plus": sum(v for k, v in counts.items() if k >= 2),
        "user_pairs_sharing_3plus": sum(v for k, v in counts.items() if k >= 3),
        "merchants_skipped_high_degree": skipped,
    }


def summarize(tx: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    """One-row structural summary of the transaction graph."""
    g = build_graph(tx)
    components = list(nx.connected_components(g))
    sizes = sorted((len(c) for c in components), reverse=True)
    degrees = dict(g.degree())
    cycles = count_four_cycles(tx)

    n_nodes, n_edges = g.number_of_nodes(), g.number_of_edges()
    users = tx["user_id_normalized"].nunique()
    merchants = tx["merchant_id_normalized"].nunique()
    pair_count = tx.groupby(["user_id_normalized", "merchant_id_normalized"]).ngroups
    # A forest has exactly (nodes - components) edges.
    is_forest = n_edges == n_nodes - len(components)

    rows = {
        "transactions": len(tx),
        "distinct_users": users,
        "distinct_merchants": merchants,
        "distinct_user_merchant_pairs": pair_count,
        "repeat_pairs": int(
            (tx.groupby(["user_id_normalized", "merchant_id_normalized"]).size() > 1).sum()
        ),
        "nodes": n_nodes,
        "edges": n_edges,
        "components": len(components),
        "largest_component_nodes": sizes[0] if sizes else 0,
        "largest_component_share_pct": round(100 * sizes[0] / n_nodes, 2) if n_nodes else 0.0,
        "mean_degree": round(sum(degrees.values()) / n_nodes, 3) if n_nodes else 0.0,
        "max_degree": max(degrees.values()) if degrees else 0,
        "mean_user_degree": round(tx.groupby("user_id_normalized")["merchant_id_normalized"].nunique().mean(), 3),
        "mean_merchant_degree": round(tx.groupby("merchant_id_normalized")["user_id_normalized"].nunique().mean(), 3),
        "is_bipartite": bool(nx.is_bipartite(g)),
        "shares_ids_across_sides": len(set(tx["user_id_normalized"]) & set(tx["merchant_id_normalized"])),
        "is_forest_acyclic": bool(is_forest),
        **cycles,
    }
    return pd.DataFrame([rows])


def component_table(tx: pd.DataFrame, cb: pd.DataFrame, top: int = 200) -> pd.DataFrame:
    """Per-component profile, ranked by dispute count then value.

    This is the closest honest analogue to 'find the suspicious ring': the most
    connected clusters that also carry disputes. They are reported as clusters
    of related activity, never as rings.
    """
    g = build_graph(tx)
    disputed = set(cb.loc[~cb["txn_unlinked"], "txn_key"].dropna())
    t = tx.assign(is_disputed=tx["txn_key"].isin(disputed))

    membership = {}
    for i, comp in enumerate(nx.connected_components(g)):
        for node in comp:
            membership[node] = i
    t = t.assign(component=t["user_id_normalized"].map(membership))

    out = t.groupby("component").agg(
        transactions=("txn_key", "size"),
        users=("user_id_normalized", "nunique"),
        merchants=("merchant_id_normalized", "nunique"),
        total_value=("amount_inr", "sum"),
        disputed_transactions=("is_disputed", "sum"),
        first_seen=("timestamp_clean", "min"),
        last_seen=("timestamp_clean", "max"),
    ).reset_index()
    out["nodes"] = out["users"] + out["merchants"]
    out["time_span_days"] = (out["last_seen"] - out["first_seen"]).dt.total_seconds() / 86400
    out["time_span_days"] = out["time_span_days"].round(1)
    out["dispute_rate_pct"] = (100 * out["disputed_transactions"] / out["transactions"]).round(2)
    # A tree component always has exactly nodes-1 edges; any extra edge would be
    # a cycle. Reported per component so the acyclicity claim is checkable.
    out["edges"] = out["transactions"]
    out["is_tree"] = out["edges"] == out["nodes"] - 1
    out = out.sort_values(["disputed_transactions", "total_value"], ascending=False)
    return out.head(top).reset_index(drop=True)


def component_edges(tx: pd.DataFrame, cb: pd.DataFrame, component_ids: list[int]) -> pd.DataFrame:
    """Edge list for the named components, for network rendering."""
    g = build_graph(tx)
    membership = {}
    for i, comp in enumerate(nx.connected_components(g)):
        for node in comp:
            membership[node] = i
    disputed = set(cb.loc[~cb["txn_unlinked"], "txn_key"].dropna())
    t = tx.assign(
        is_disputed=tx["txn_key"].isin(disputed),
        component=tx["user_id_normalized"].map(membership),
    )
    sub = t[t["component"].isin(component_ids)]
    return sub[[
        "component", "user_id_normalized", "merchant_id_normalized", "txn_key",
        "amount_inr", "status_canonical", "is_disputed", "timestamp_clean",
    ]].reset_index(drop=True)
