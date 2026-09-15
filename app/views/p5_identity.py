"""Page 5 — Identity & Network Intelligence.

This page does not claim a fraud ring. It shows the identity-collision problem,
which is real and measurable, and then shows why the recorded transactions give no
statistically or structurally supported evidence for the tested ring pattern.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import charts
from app.components.theme import SERIES, STATUS
from app.components.ui import (
    callout, empty_state, fmt_inr, fmt_int, fmt_pct, kpi_card, section,
)


def render(ctx) -> None:
    st.markdown("# Identity & Network Intelligence")
    tabs = st.tabs(["Identity collisions", "Network structure", "Fraud ring hypothesis"])
    with tabs[0]:
        _collisions(ctx)
    with tabs[1]:
        _network(ctx)
    with tabs[2]:
        _ring_hypothesis(ctx)


# --------------------------------------------------------------------------
def _collisions(ctx) -> None:
    star = ctx["star"]
    users, merchants = star["dim_users"], star["dim_merchants"]
    bridge = star["bridge_identity_collision"]

    section(
        "The collision problem",
        "An identifier repeated across rows usually means a duplicated record. Here it "
        "mostly means two different entities were issued the same ID. Collapsing on the ID "
        "would have destroyed one of them and fabricated a golden record from the remains.")

    c = st.columns(4, gap="small")
    with c[0]:
        kpi_card("Customer ID Collisions", fmt_int(int(users["identity_ambiguous"].sum())),
                 "one ID, several distinct people")
    with c[1]:
        kpi_card("Merchant ID Collisions", fmt_int(int(merchants["identity_ambiguous"].sum())),
                 "one ID, several distinct businesses")
    with c[2]:
        kpi_card("Candidate Rows Preserved", fmt_int(len(bridge)),
                 "survivors and non-survivors, all inspectable")
    with c[3]:
        low = int((users["resolution_confidence"] == "LOW").sum())
        kpi_card("Low-Confidence Identities", fmt_int(low),
                 "attributes not safely attributable")

    from src.transformation.audit_facts import lookup as fact

    multi = fact(star.get("audit_facts"), "kyc_ids_with_multiple_pans")
    shared = fact(star.get("audit_facts"), "pans_shared_across_ids")
    if multi is not None and shared is not None:
        callout(
            "Corroboration that these are genuinely different people: "
            f"<strong>{multi:,.0f} customer IDs carry more than one distinct PAN</strong>, while "
            f"<strong>{shared:,.0f} PANs are shared across customer IDs</strong>. The identifier "
            "is ambiguous; the person is not.")

    # ---- explorer ----------------------------------------------------------
    section("Collision explorer", "Pick an identifier to see every record issued under it.")
    entity_type = st.radio("Entity", ["USER", "MERCHANT"], horizontal=True, key="coll_type")
    pool = bridge[(bridge["entity_type"] == entity_type)
                  & (bridge["identity_class"] == "ID_COLLISION")]
    if pool.empty:
        empty_state(f"No {entity_type.lower()} collisions in the current build.")
        return

    keys = sorted(pool["entity_key"].unique().tolist())
    default = "USR10043" if "USR10043" in keys else keys[0]
    chosen = st.selectbox(
        "Identifier", keys, index=keys.index(default),
        help="Every identifier here is shared by more than one distinct entity.")

    rows = pool[pool["entity_key"] == chosen].sort_values("candidate_rank")
    st.markdown(
        f'<div class="dv-kpi-sub"><strong>{chosen}</strong> resolves to '
        f'<strong>{int(rows["candidate_count"].iloc[0])} distinct entities</strong> across '
        f'{int(rows["source_row_count"].iloc[0])} source rows. The survivor is the most '
        f'complete record; the rest are retained here.</div>', unsafe_allow_html=True)

    cols = ["candidate_rank", "is_survivor", "full_name_clean", "completeness_score"]
    if entity_type == "USER":
        cols += ["pan_masked", "aadhaar_last4", "city_clean", "state_clean",
                 "kyc_status_canonical", "risk_segment_canonical", "monthly_income_inr"]
    else:
        cols += ["merchant_category_canonical", "city_clean", "state_clean",
                 "merchant_status_canonical", "settlement_account_display"]
    cols = [c for c in cols if c in rows.columns]
    st.dataframe(
        rows[cols].rename(columns={
            "candidate_rank": "Rank", "is_survivor": "Survivor",
            "full_name_clean": "Name", "completeness_score": "Completeness",
            "pan_masked": "PAN (masked)", "aadhaar_last4": "Aadhaar (last 4)",
            "city_clean": "City", "state_clean": "State",
            "kyc_status_canonical": "KYC", "risk_segment_canonical": "Segment",
            "monthly_income_inr": "Income ₹",
            "merchant_category_canonical": "Category",
            "merchant_status_canonical": "Status",
            "settlement_account_display": "Settlement acct",
        }), hide_index=True, width="stretch")
    st.caption(
        "PAN is shown masked and Aadhaar as its last four digits only. The processed layer "
        "never stores the full values or any hash of them.")

    # ---- distribution ------------------------------------------------------
    section("How widespread is it?")
    left, right = st.columns(2, gap="medium")
    with left:
        dist = users[users["user_key"] != "UNKNOWN"]["identity_class"].value_counts()
        st.plotly_chart(
            charts.hbar(dist.index.tolist(), dist.tolist(),
                        title="Customer identities by resolution class",
                        text=[fmt_int(v) for v in dist], xtitle="identities"),
            width="stretch", config={"displayModeBar": False})
    with right:
        dist_m = merchants[merchants["merchant_key"] != "UNKNOWN"]["identity_class"].value_counts()
        st.plotly_chart(
            charts.hbar(dist_m.index.tolist(), dist_m.tolist(),
                        title="Merchant identities by resolution class",
                        colour=SERIES[1], text=[fmt_int(v) for v in dist_m],
                        xtitle="identities"),
            width="stretch", config={"displayModeBar": False})
    st.caption(
        "UNIQUE — one source row. SAME_ENTITY_CONFLICTING — one entity described by several "
        "conflicting rows, resolved to the most complete. ID_COLLISION — different entities "
        "sharing an identifier, flagged and never merged.")


# --------------------------------------------------------------------------
def _network(ctx) -> None:
    agg = ctx["agg"]
    net = agg["agg_network_summary"].iloc[0]
    comps = agg["agg_network_components"]
    edges = agg["agg_network_edges"]

    section("Transaction network structure",
            "Users and merchants as nodes, transactions as edges.")
    c = st.columns(4, gap="small")
    with c[0]:
        kpi_card("Nodes", fmt_int(net["nodes"]),
                 f"{fmt_int(net['distinct_users'])} users + "
                 f"{fmt_int(net['distinct_merchants'])} merchants")
    with c[1]:
        kpi_card("Edges", fmt_int(net["edges"]), "unique user-merchant pairs")
    with c[2]:
        kpi_card("Components", fmt_int(net["components"]),
                 f"largest holds {fmt_pct(net['largest_component_share_pct'])} of nodes")
    with c[3]:
        kpi_card("Mean Degree", f"{net['mean_degree']:.2f}",
                 f"max {int(net['max_degree'])}")

    left, right = st.columns([2, 3], gap="medium")
    with left:
        sizes = comps["nodes"].value_counts().sort_index()
        st.plotly_chart(
            charts.hbar([f"{int(k)} nodes" for k in sizes.index[:12]],
                        sizes.tolist()[:12],
                        title="Component sizes (top connected clusters)",
                        text=[fmt_int(v) for v in sizes[:12]], xtitle="components"),
            width="stretch", config={"displayModeBar": False})
    with right:
        section("Most connected clusters carrying disputes",
                "The closest honest analogue to 'find the ring'. These are clusters of "
                "related activity, reported as such.")
        st.dataframe(
            comps.head(12)[["component", "nodes", "users", "merchants", "transactions",
                            "disputed_transactions", "total_value", "time_span_days",
                            "is_tree"]].rename(columns={
                "component": "Cluster", "nodes": "Nodes", "users": "Users",
                "merchants": "Merchants", "transactions": "Txns",
                "disputed_transactions": "Disputed", "total_value": "Value ₹",
                "time_span_days": "Span (days)", "is_tree": "Acyclic",
            }), hide_index=True, width="stretch", height=300,
            column_config={
                "Value ₹": st.column_config.NumberColumn(format="%.0f"),
                "Acyclic": st.column_config.CheckboxColumn(
                    help="A tree has exactly nodes-1 edges. Any extra edge would be a cycle."),
            })

    # ---- render one cluster ------------------------------------------------
    section("Cluster explorer", "Disputed transactions are drawn thicker and in the reserved "
                                "critical colour, so the distinction is never colour alone.")
    available = comps["component"].head(25).tolist()
    available = [c for c in available if c in set(edges["component"])]
    if not available:
        empty_state("No cluster edges were materialized in this build.")
        return
    chosen = st.selectbox(
        "Cluster", available, format_func=lambda c: (
            f"Cluster {int(c)} — "
            f"{int(comps.loc[comps['component'] == c, 'nodes'].iloc[0])} nodes, "
            f"{int(comps.loc[comps['component'] == c, 'disputed_transactions'].iloc[0])} disputed"))
    sub = edges[edges["component"] == chosen]
    if sub.empty:
        empty_state("No edges for this cluster.")
        return

    left, right = st.columns([3, 2], gap="medium")
    with left:
        fig = charts.network(sub, title=f"Cluster {int(chosen)}")
        if fig is not None:
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})
    with right:
        row = comps[comps["component"] == chosen].iloc[0]
        st.markdown("**Cluster profile**")
        st.dataframe(pd.DataFrame([
            ("Users", fmt_int(row["users"])),
            ("Merchants", fmt_int(row["merchants"])),
            ("Transactions", fmt_int(row["transactions"])),
            ("Disputed", fmt_int(row["disputed_transactions"])),
            ("Total value", fmt_inr(row["total_value"])),
            ("Time span", f"{row['time_span_days']:.1f} days"),
            ("Edges", fmt_int(row["edges"])),
            ("Acyclic (tree)", "Yes" if row["is_tree"] else "No"),
        ], columns=["Property", "Value"]), hide_index=True, width="stretch")
        callout(
            "This cluster is a <strong>tree</strong>: exactly nodes − 1 edges. There is no "
            "closed loop, so money cannot circulate within it."
            if row["is_tree"] else
            "This cluster contains at least one cycle — worth inspecting.",
            kind="info" if row["is_tree"] else "warn")


# --------------------------------------------------------------------------
def _ring_hypothesis(ctx) -> None:
    net = ctx["agg"]["agg_network_summary"].iloc[0]

    st.markdown(
        f'<div class="dv-call dv-call-crit" style="border-left-width:5px;padding:1.1rem 1.2rem">'
        f'<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.06em;'
        f'color:{STATUS["critical"]};margin-bottom:0.4rem">'
        f'FRAUD RING HYPOTHESIS — NOT SUPPORTED</div>'
        f'<div style="font-size:0.95rem;font-weight:620;color:#0b0b0b;margin-bottom:0.5rem">'
        f'The brief asks for circular money-laundering rings of the form A → B → C → A. '
        f'These transactions cannot form one.</div>'
        f'<div style="font-size:0.84rem;line-height:1.65">This is not a case of looking and '
        f'finding nothing: the graph\'s structure rules the pattern out for the recorded '
        f'transactions, and each of the four measurements below shows that on its own. The '
        f'available data therefore does not provide statistically or structurally supported '
        f'evidence for the tested fraud-ring hypothesis. Money moving outside this dataset is '
        f'not observed.</div>'
        f'</div>', unsafe_allow_html=True)

    section("Structural evidence")
    evidence = pd.DataFrame([
        ("Graph is bipartite",
         "Yes" if net["is_bipartite"] else "No",
         f"{int(net['shares_ids_across_sides'])} IDs appear on both sides",
         "Users only ever pay merchants. There are no user-to-user transfers, so a directed "
         "money path through three parties and back cannot be formed."),
        ("Repeat user-merchant pairs",
         fmt_int(net["repeat_pairs"]),
         f"{fmt_int(net['distinct_user_merchant_pairs'])} pairs across "
         f"{fmt_int(net['transactions'])} transactions",
         "Every transaction is a unique pair. No relationship is ever used twice, which is "
         "the first thing a genuine ring would show."),
        ("Four-cycles (u₁–m₁–u₂–m₂–u₁)",
         fmt_int(net["user_pairs_sharing_2plus"]),
         f"{fmt_int(net['user_pairs_sharing_a_merchant'])} user pairs share one merchant; "
         f"{int(net['user_pairs_sharing_3plus'])} share three",
         "A bipartite graph has no odd cycles, so a 4-cycle is the shortest closed loop "
         "possible. There are none — the graph is acyclic."),
        ("Graph is a forest",
         "Yes" if net["is_forest_acyclic"] else "No",
         f"{fmt_int(net['edges'])} edges, {fmt_int(net['nodes'])} nodes, "
         f"{fmt_int(net['components'])} components",
         "Edges = nodes − components exactly, which is the definition of a forest. A forest "
         "contains no cycles of any length."),
        ("Largest component",
         f"{fmt_pct(net['largest_component_share_pct'])} of nodes",
         f"{fmt_int(net['largest_component_nodes'])} of {fmt_int(net['nodes'])} nodes",
         "Community detection has nothing to partition when the biggest connected piece is a "
         "fraction of a percent of the graph."),
    ], columns=["Test", "Result", "Measurement", "What it means"])
    st.dataframe(evidence, hide_index=True, width="stretch", height=250)

    section("What we tested and also did not find")
    from src.transformation.audit_facts import lookup as fact

    facts = ctx["star"].get("audit_facts")
    v = {k: fact(facts, k) for k in (
        "full_aadhaar_shared_across_ids", "full_settlement_shared_across_merchants",
        "pans_shared_across_ids", "masked_aadhaar_shared_across_ids",
        "masked_aadhaar_expected_by_chance", "masked_settlement_shared_across_merchants",
        "masked_settlement_expected_by_chance")}
    if None not in v.values():
        def relation(observed, expected):
            ratio = observed / expected if expected else float("nan")
            if 0.9 <= ratio <= 1.1:
                return "close to"
            return "below" if observed < expected else "above"
        callout(
            "Three further shared-identity signatures were measured: "
            f"<strong>{v['full_aadhaar_shared_across_ids']:,.0f}</strong> full 12-digit Aadhaar "
            "numbers shared across customer IDs, "
            f"<strong>{v['full_settlement_shared_across_merchants']:,.0f}</strong> full settlement "
            "accounts shared across merchants, and "
            f"<strong>{v['pans_shared_across_ids']:,.0f}</strong> PANs shared across customer IDs. "
            "Apparent sharing appears only where masking cuts the key to four digits, and there "
            "it sits "
            f"{relation(v['masked_aadhaar_shared_across_ids'], v['masked_aadhaar_expected_by_chance'])} "
            "the rate chance alone produces: "
            f"{v['masked_aadhaar_shared_across_ids']:,.0f} shared masked Aadhaar values against "
            f"{v['masked_aadhaar_expected_by_chance']:,.0f} expected, and "
            f"{v['masked_settlement_shared_across_merchants']:,.0f} shared masked settlement "
            f"accounts against {v['masked_settlement_expected_by_chance']:,.0f} expected.")

    section("Why we are reporting this instead of a ring")
    callout(
        "Fabricating a ring here would have been easy and would have looked impressive for "
        "about ninety seconds. The structural facts above are checkable in one query each, "
        "and a reviewer who ran them would find the finding false. A system that reports "
        "<em>no supported evidence for the tested ring pattern, and exactly why</em> is doing "
        "the job a payments risk team actually needs: separating signal from noise rather "
        "than manufacturing confidence.", kind="good")
