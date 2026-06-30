import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Training Dashboard", layout="wide", page_icon="🏃")

DATA_PATH = Path(__file__).parent / "garmin" / "data.json"


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    activities = []
    for entries in raw.get("activities", {}).values():
        activities.extend(entries if isinstance(entries, list) else [entries])
    wellness = list(raw.get("wellness", {}).values())
    return activities, wellness


def build_act_df(activities):
    rows = []
    for a in activities:
        d_km = (a.get("distance_m") or 0) / 1000
        dur_s = a.get("duration_s") or 0
        pace = (dur_s / 60) / d_km if d_km > 0.1 else None
        rows.append({
            "date": pd.to_datetime(a["date"]),
            "name": a.get("name", "Activity"),
            "type": (a.get("type") or "unknown").lower(),
            "distance_km": round(d_km, 2),
            "duration_min": round(dur_s / 60, 1),
            "pace_min_km": round(pace, 3) if pace else None,
            "avg_hr": a.get("avg_hr"),
            "max_hr": a.get("max_hr"),
            "calories": a.get("calories"),
            "vo2max": a.get("vo2max"),
            "aerobic_te": a.get("training_effect_aerobic"),
            "anaerobic_te": a.get("training_effect_anaerobic"),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["run_type"] = df.apply(
        lambda r: classify_run(r["distance_km"], r["aerobic_te"], r["duration_min"])
        if r["type"] == "running" else r["type"].title(),
        axis=1,
    )
    return df


def build_well_df(wellness):
    rows = []
    for w in wellness:
        rows.append({
            "date": pd.to_datetime(w["date"]),
            "steps": w.get("steps"),
            "stress_avg": w.get("stress_avg"),
            "bb_low": w.get("body_battery_low"),
            "bb_high": w.get("body_battery_high"),
            "sleep_hours": w.get("sleep_hours"),
            "sleep_score": w.get("sleep_score"),
            "hrv": w.get("hrv"),
            "resting_hr": w.get("resting_hr"),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["bb_recovery"] = df["bb_high"] - df["bb_low"]
    return df


# ── Maths helpers ─────────────────────────────────────────────────────────────
def pace_str(min_per_km):
    if not min_per_km or pd.isna(min_per_km):
        return "—"
    m = int(min_per_km)
    s = int(round((min_per_km - m) * 60))
    return f"{m}:{s:02d} /km"


def seconds_to_hms(total_seconds):
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def riegel_predict(t1_min, d1_km, d2_km=21.0975):
    """Riegel endurance formula."""
    return t1_min * (d2_km / d1_km) ** 1.06


def daniels_pct_vo2max(t_min):
    """Fraction of VO2max used at race pace for a given duration (Daniels)."""
    return (0.8 + 0.1894393 * math.exp(-0.012778 * t_min)
            + 0.2989558 * math.exp(-0.1932605 * t_min))


def daniels_v_from_vo2(vo2):
    """Speed in m/min from VO2 using Daniels polynomial."""
    a, b, c = 0.000104, 0.182258, -4.60 - vo2
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return None
    return (-b + math.sqrt(disc)) / (2 * a)


def vo2max_to_hm_prediction(vo2max, dist_km=21.0975):
    """Newton-iterate to find race time that satisfies Daniels equations."""
    dist_m = dist_km * 1000
    t = 90.0  # initial guess minutes
    for _ in range(50):
        pct = daniels_pct_vo2max(t)
        v = daniels_v_from_vo2(vo2max * pct)
        if not v:
            break
        t_new = dist_m / v
        if abs(t_new - t) < 0.01:
            t = t_new
            break
        t = t_new
    return t  # minutes


def training_paces(vo2max):
    """Return Jack Daniels training paces (min/km) for key zones."""
    pcts = {"Easy": 0.65, "Marathon": 0.83, "Threshold": 0.88, "Interval": 0.975}
    paces = {}
    for zone, pct in pcts.items():
        v = daniels_v_from_vo2(vo2max * pct)
        if v:
            paces[zone] = 1000 / v  # min per km
    return paces


# ── Run classification ────────────────────────────────────────────────────────
def classify_run(dist_km, aerobic_te, duration_min):
    """
    Classify a run using Garmin's aerobic training effect + distance.
    TE 1-2 = recovery/base, 2-3 = base/easy, 3-4 = improving, 4-5 = hard/tempo.
    Long run overrides effort label for distance >= 14 km.
    """
    if dist_km >= 14:
        return "Long Run"
    if aerobic_te is None:
        if dist_km >= 10:
            return "Long Run"
        return "Easy"
    if aerobic_te >= 4.0:
        return "Tempo / Hard"
    if aerobic_te >= 3.0:
        return "Moderate"
    return "Easy / Recovery"


RUN_TYPE_COLORS = {
    "Long Run":       "#f59e0b",
    "Tempo / Hard":   "#ef4444",
    "Moderate":       "#3b82f6",
    "Easy / Recovery": "#10b981",
}


# ── Calendar heatmap ──────────────────────────────────────────────────────────
def make_calendar_heatmap(act_df):
    if act_df.empty:
        return None
    start = act_df["date"].min()
    end = act_df["date"].max()
    all_dates = pd.date_range(start, end)
    daily_km = act_df.groupby("date")["distance_km"].sum().reindex(all_dates, fill_value=0)

    # Build week×day matrix
    weeks, days = [], []
    for d, km in daily_km.items():
        weeks.append(d.isocalendar()[1])
        days.append(d.weekday())  # 0=Mon

    df_cal = pd.DataFrame({"date": all_dates, "km": daily_km.values,
                            "week": [d.isocalendar()[1] + (d.year - start.year) * 52
                                     for d in all_dates],
                            "dow": [d.weekday() for d in all_dates]})

    pivot = df_cal.pivot_table(index="dow", columns="week", values="km", aggfunc="sum").fillna(0)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    week_starts = df_cal.groupby("week")["date"].min()
    col_labels = [d.strftime("%d %b") if i % 4 == 0 else "" for i, d in enumerate(week_starts)]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(range(len(pivot.columns))),
        y=day_labels,
        colorscale=[[0, "#1e293b"], [0.01, "#1e3a5f"], [1, "#3b82f6"]],
        showscale=False,
        hovertemplate="<b>%{text}</b><br>%{z:.1f} km<extra></extra>",
        text=[[week_starts.iloc[c].strftime("%d %b") if c < len(week_starts) else ""
               for c in range(len(pivot.columns))]
              for _ in range(7)],
    ))
    fig.update_layout(
        margin=dict(t=5, b=5, l=40, r=5),
        height=160,
        xaxis=dict(tickvals=list(range(len(col_labels))), ticktext=col_labels, tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="white",
    )
    return fig


# ── Rule-based analysis (no API required) ────────────────────────────────────
def auto_analysis(act_df, well_df, focus="general"):
    runs = act_df[act_df["type"] == "running"].sort_values("date")
    if runs.empty:
        return "No running data found."

    well_recent = well_df.tail(30)
    weekly = runs.copy()
    weekly["week"] = weekly["date"].dt.to_period("W")
    wk_km = weekly.groupby("week")["distance_km"].sum()

    total_km = runs["distance_km"].sum()
    avg_wk = wk_km.mean()
    peak_wk = wk_km.max()
    total_runs = len(runs)
    weeks_active = wk_km[wk_km > 0].count()
    total_weeks = len(wk_km)
    consistency_pct = round(weeks_active / total_weeks * 100) if total_weeks else 0

    early_runs = runs.head(max(1, len(runs) // 3))
    late_runs = runs.tail(max(1, len(runs) // 3))

    vo2_early = early_runs["vo2max"].dropna().mean() if early_runs["vo2max"].notna().any() else None
    vo2_late = late_runs["vo2max"].dropna().mean() if late_runs["vo2max"].notna().any() else None
    pace_early = early_runs["pace_min_km"].dropna().mean() if early_runs["pace_min_km"].notna().any() else None
    pace_late = late_runs["pace_min_km"].dropna().mean() if late_runs["pace_min_km"].notna().any() else None
    hr_early = early_runs["avg_hr"].dropna().mean() if early_runs["avg_hr"].notna().any() else None
    hr_late = late_runs["avg_hr"].dropna().mean() if late_runs["avg_hr"].notna().any() else None
    longest = runs["distance_km"].max()

    avg_bb_high = well_recent["bb_high"].dropna().mean() if well_recent["bb_high"].notna().any() else None
    avg_bb_rec = well_recent["bb_recovery"].dropna().mean() if well_recent["bb_recovery"].notna().any() else None
    avg_stress = well_recent["stress_avg"].dropna().mean() if well_recent["stress_avg"].notna().any() else None

    paras = []

    # ── Paragraph 1: Training load ────────────────────────────────────────────
    if consistency_pct >= 80:
        consistency_word = "highly consistent"
    elif consistency_pct >= 60:
        consistency_word = "reasonably consistent"
    else:
        consistency_word = "inconsistent"

    load_note = (f"Your training has been {consistency_word} over the last 6 months — "
                 f"you ran in {weeks_active} of {total_weeks} weeks ({consistency_pct}%), "
                 f"covering {total_km:.0f} km across {total_runs} runs. "
                 f"Your average week sits at {avg_wk:.1f} km, "
                 f"with a peak of {peak_wk:.1f} km. ")

    if avg_wk < 20:
        load_note += ("Volume is on the lower side for half marathon preparation — "
                      "most coaches recommend 40–60 km/week in a build phase.")
    elif avg_wk < 40:
        load_note += ("Volume is solid for a recreational runner. "
                      "To target a competitive half marathon, a gradual build toward 50+ km/week would help.")
    else:
        load_note += ("Volume is strong. Make sure your easy days are genuinely easy to support recovery.")

    paras.append(load_note)

    # ── Paragraph 2: Fitness trajectory ──────────────────────────────────────
    fitness_parts = []
    if vo2_early and vo2_late:
        diff = vo2_late - vo2_early
        if diff > 1.5:
            fitness_parts.append(f"VO2max has improved from {vo2_early:.0f} to {vo2_late:.0f} — "
                                  f"a meaningful {diff:.1f}-point gain that reflects real aerobic development.")
        elif diff < -1.5:
            fitness_parts.append(f"VO2max has dipped from {vo2_early:.0f} to {vo2_late:.0f}. "
                                  f"This can happen after a break or during a high-stress period.")
        else:
            fitness_parts.append(f"VO2max has held steady around {vo2_late:.0f}, "
                                  f"suggesting maintenance rather than a fitness build.")

    if pace_early and pace_late:
        pace_diff = pace_early - pace_late
        if pace_diff > 0.1:
            fitness_parts.append(f"Average pace has improved by {pace_diff*60:.0f} seconds/km "
                                  f"({pace_str(pace_early)} → {pace_str(pace_late)}).")
        elif pace_diff < -0.1:
            fitness_parts.append(f"Average pace has slowed by {abs(pace_diff)*60:.0f} seconds/km. "
                                  f"Check if recent runs are higher effort or longer distance.")
        else:
            fitness_parts.append(f"Average pace is consistent at around {pace_str(pace_late)}.")

    if hr_early and hr_late and pace_early and pace_late:
        hr_diff = hr_late - hr_early
        pace_diff = pace_early - pace_late
        if hr_diff < -3 and pace_diff >= 0:
            fitness_parts.append("Running at the same or faster pace with a lower HR — "
                                  "a clear sign of improving aerobic efficiency.")
        elif hr_diff > 3 and pace_diff < 0:
            fitness_parts.append("HR is higher and pace slower recently — "
                                  "could indicate accumulated fatigue or reduced fitness.")

    paras.append(" ".join(fitness_parts) if fitness_parts else
                 "Not enough data to compare early vs recent fitness.")

    # ── Paragraph 3: Recovery ─────────────────────────────────────────────────
    rec_parts = []
    if avg_bb_high is not None:
        if avg_bb_high >= 70:
            rec_parts.append(f"Your body battery peaks at {avg_bb_high:.0f} on average — "
                              "excellent recovery. You're absorbing your training well.")
        elif avg_bb_high >= 50:
            rec_parts.append(f"Body battery peaks at {avg_bb_high:.0f} — adequate but "
                              "there's room to improve. More sleep or easier days would help.")
        else:
            rec_parts.append(f"Body battery peaking at only {avg_bb_high:.0f} is a warning sign. "
                              "Your body isn't fully recovering between sessions.")

    if avg_bb_rec is not None:
        if avg_bb_rec >= 50:
            rec_parts.append(f"Overnight recovery is strong at {avg_bb_rec:.0f} points gained.")
        elif avg_bb_rec >= 30:
            rec_parts.append(f"Overnight recovery averages {avg_bb_rec:.0f} points — moderate.")
        else:
            rec_parts.append(f"Only recovering {avg_bb_rec:.0f} points overnight suggests "
                              "sleep quality or duration needs attention.")

    if avg_stress is not None:
        if avg_stress < 30:
            rec_parts.append(f"Stress is low at {avg_stress:.0f}/100 — good conditions for adaptation.")
        elif avg_stress < 50:
            rec_parts.append(f"Stress averages {avg_stress:.0f}/100 — manageable, but monitor it.")
        else:
            rec_parts.append(f"Stress at {avg_stress:.0f}/100 is elevated. "
                              "High chronic stress blunts training adaptation.")

    paras.append(" ".join(rec_parts) if rec_parts else "Recovery data looks good.")

    # ── Paragraph 4: Recommendation ──────────────────────────────────────────
    if focus == "race":
        if longest < 16:
            rec = (f"Your longest run is {longest:.1f} km. For a half marathon, "
                   "you need at least one run of 18–20 km before race day. "
                   "Build your long run by 1–2 km each week until you hit 18 km, "
                   "then taper for 2 weeks.")
        elif avg_wk < 35:
            rec = (f"Averaging {avg_wk:.1f} km/week is workable but thin for a half. "
                   "Try adding one extra easy run (8–10 km) per week for the next 6 weeks "
                   "to build your aerobic base before the taper.")
        elif avg_bb_high is not None and avg_bb_high < 55:
            rec = ("Your recovery markers suggest you're carrying fatigue. "
                   "Before adding more volume, prioritise sleep and add one full rest day. "
                   "Under-recovered athletes plateau — more isn't always better.")
        else:
            rec = ("Your base looks solid. The next step is one weekly threshold session: "
                   f"20 minutes at {pace_str(pace_late * 0.92 if pace_late else None)} "
                   "(comfortably hard, not all-out). "
                   "This is the single biggest lever for half marathon performance.")
    else:
        if consistency_pct < 60:
            rec = ("Your biggest opportunity is consistency. Missing weeks means losing fitness faster "
                   "than you build it. Aim to run at least 3 times every week for the next 8 weeks, "
                   "even if some runs are short.")
        elif avg_wk < 30:
            rec = (f"At {avg_wk:.1f} km/week, a gradual volume increase is the highest-value move. "
                   "Add 10% per week for 3 weeks, then take an easier week. "
                   "Don't add speed work until you're consistently above 40 km/week.")
        elif avg_bb_high is not None and avg_bb_high < 55:
            rec = (f"Body battery averaging {avg_bb_high:.0f} suggests your recovery isn't keeping pace "
                   "with your training. Prioritise sleep quality — even 30 extra minutes per night "
                   "has measurable impact on body battery and training adaptation.")
        else:
            late_vo2 = runs["vo2max"].dropna().iloc[-1] if runs["vo2max"].notna().any() else None
            if late_vo2:
                paces = training_paces(late_vo2)
                thresh = paces.get("Threshold")
                rec = (f"Your aerobic base is solid. Add one threshold session per week: "
                       f"3×8 minutes at {pace_str(thresh)} with 3 minutes recovery. "
                       "This is the most time-efficient way to push your VO2max and race pace higher.")
            else:
                rec = ("Your training looks well-rounded. The next step is adding structured quality — "
                       "one tempo run per week at comfortably hard effort will unlock the next level.")

    paras.append(f"**Recommendation:** {rec}")
    return "\n\n".join(paras)


# ── Main ──────────────────────────────────────────────────────────────────────
activities, wellness = load_data()
act_df = build_act_df(activities)
well_df = build_well_df(wellness)
runs_df = act_df[act_df["type"] == "running"].copy()

weekly_agg = (runs_df.groupby(runs_df["date"].dt.to_period("W").dt.start_time)
              .agg(volume_km=("distance_km", "sum"),
                   hours=("duration_min", lambda x: x.sum() / 60),
                   count=("date", "count"))
              .reset_index().rename(columns={"date": "week"}))
weekly_agg["volume_km"] = weekly_agg["volume_km"].round(1)
weekly_agg["hours"] = weekly_agg["hours"].round(1)

# Header
st.title("🏃 Training Dashboard")
if not act_df.empty:
    col1, col2, col3, col4, col5 = st.columns(5)
    wk = weekly_agg.tail(1)
    col1.metric("This week", f"{wk['volume_km'].values[0]:.1f} km" if not wk.empty else "—")
    col2.metric("4-wk avg", f"{weekly_agg.tail(4)['volume_km'].mean():.1f} km/wk")
    col3.metric("Total runs", f"{len(runs_df)}")
    col4.metric("VO2max", f"{runs_df['vo2max'].dropna().iloc[-1]:.0f}" if runs_df["vo2max"].notna().any() else "—")
    col5.metric("Body battery now", f"{well_df['bb_high'].dropna().iloc[-1]:.0f}" if well_df["bb_high"].notna().any() else "—")

st.divider()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📅 Overview", "⚡ Performance", "💚 Recovery", "🏁 Race Predictor", "🤖 AI Coach"])


# ── TAB 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Weekly volume (km)")
        fig = px.bar(weekly_agg, x="week", y="volume_km", color_discrete_sequence=["#3b82f6"])
        fig.update_layout(margin=dict(t=5, b=0), xaxis_title="", yaxis_title="km", height=260)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Weekly hours")
        fig = px.bar(weekly_agg, x="week", y="hours", color_discrete_sequence=["#10b981"])
        fig.update_layout(margin=dict(t=5, b=0), xaxis_title="", yaxis_title="hours", height=260)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Training calendar")
    cal_fig = make_calendar_heatmap(act_df)
    if cal_fig:
        st.plotly_chart(cal_fig, use_container_width=True)
        st.caption("Darker blue = more km that day.")

    st.subheader("All activities")
    if not act_df.empty:
        disp = act_df.sort_values("date", ascending=False).copy()
        disp["Date"] = disp["date"].dt.strftime("%d %b %Y")
        disp["Distance"] = disp["distance_km"].apply(lambda d: f"{d:.2f} km")
        disp["Duration"] = disp["duration_min"].apply(
            lambda m: f"{int(m//60)}h {int(m%60):02d}m" if m >= 60 else f"{int(m)}m")
        disp["Pace"] = disp["pace_min_km"].apply(pace_str)
        disp["Avg HR"] = disp["avg_hr"].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "—")
        disp["VO2max"] = disp["vo2max"].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "—")
        st.dataframe(
            disp[["Date", "name", "run_type", "Distance", "Duration", "Pace", "Avg HR", "VO2max"]]
            .rename(columns={"name": "Activity", "run_type": "Type"}),
            use_container_width=True, hide_index=True, height=400)


# ── TAB 2: Performance ────────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Pace over time")
        pace_data = runs_df.dropna(subset=["pace_min_km"])
        pace_data = pace_data[pace_data["pace_min_km"] < 10].copy()
        if not pace_data.empty:
            fig = px.scatter(
                pace_data, x="date", y="pace_min_km",
                color="run_type",
                color_discrete_map=RUN_TYPE_COLORS,
                hover_data={"name": True, "distance_km": True, "run_type": False},
                labels={"pace_min_km": "Pace (min/km)", "date": "", "run_type": "Type"},
                category_orders={"run_type": ["Easy / Recovery", "Moderate", "Long Run", "Tempo / Hard"]},
                size_max=10,
            )
            # Rolling average across all runs
            rolling = (pace_data.set_index("date")["pace_min_km"]
                       .rolling("28D").mean().reset_index())
            fig.add_trace(go.Scatter(
                x=rolling["date"], y=rolling["pace_min_km"],
                mode="lines", line=dict(color="white", width=2, dash="dash"),
                name="28-day avg", showlegend=True,
            ))
            fig.update_layout(margin=dict(t=5, b=0), xaxis_title="",
                              yaxis_title="min/km", height=280, yaxis_autorange="reversed",
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Lower = faster. White dashed line = 28-day rolling average across all runs.")

    with col_b:
        st.subheader("VO2max trend")
        vo2_data = runs_df.dropna(subset=["vo2max"])
        if not vo2_data.empty:
            fig = px.line(vo2_data, x="date", y="vo2max",
                          markers=True, color_discrete_sequence=["#8b5cf6"])
            fig.update_traces(marker=dict(size=8))
            fig.update_layout(margin=dict(t=5, b=0), xaxis_title="",
                              yaxis_title="VO2max", height=280)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("HR vs Pace — by run type")
    hr_pace = runs_df.dropna(subset=["avg_hr", "pace_min_km"])
    hr_pace = hr_pace[hr_pace["pace_min_km"] < 10].copy()
    if not hr_pace.empty:
        fig = px.scatter(
            hr_pace, x="pace_min_km", y="avg_hr",
            color="run_type",
            size="distance_km",
            color_discrete_map=RUN_TYPE_COLORS,
            hover_data={"date": True, "distance_km": True, "name": True, "run_type": False},
            labels={"pace_min_km": "Pace (min/km)", "avg_hr": "Avg HR (bpm)",
                    "run_type": "Type"},
            size_max=18,
            category_orders={"run_type": ["Easy / Recovery", "Moderate", "Long Run", "Tempo / Hard"]},
        )
        fig.update_layout(margin=dict(t=5, b=0), height=340,
                          legend=dict(title="Run type", orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Easy runs cluster bottom-right (slow + low HR). Tempo top-left (fast + high HR). "
                   "Aerobic efficiency improves when easy runs shift left over time. Bubble size = distance.")

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Aerobic training effect")
        te_data = runs_df.dropna(subset=["aerobic_te"])
        if not te_data.empty:
            labels = {1: "Recovery", 2: "Base", 3: "Improving", 4: "Highly improving", 5: "Overreaching"}
            te_data = te_data.copy()
            te_data["TE band"] = te_data["aerobic_te"].apply(lambda v: labels.get(int(v), "Unknown"))
            counts = te_data["TE band"].value_counts().reset_index()
            counts.columns = ["Band", "Runs"]
            fig = px.pie(counts, names="Band", values="Runs",
                         color_discrete_sequence=px.colors.sequential.Blues_r,
                         hole=0.45)
            fig.update_layout(margin=dict(t=5, b=0), height=280, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Run distance distribution")
        if not runs_df.empty:
            fig = px.histogram(runs_df, x="distance_km", nbins=12,
                               color_discrete_sequence=["#3b82f6"],
                               labels={"distance_km": "Distance (km)"})
            fig.update_layout(margin=dict(t=5, b=0), height=280,
                              yaxis_title="Runs", bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)


# ── TAB 3: Recovery ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Body battery — daily range")
    bb_data = well_df.dropna(subset=["bb_low", "bb_high"])
    if not bb_data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bb_data["date"], y=bb_data["bb_high"],
            mode="lines", line=dict(color="#10b981", width=0),
            name="High", fill=None,
        ))
        fig.add_trace(go.Scatter(
            x=bb_data["date"], y=bb_data["bb_low"],
            mode="lines", line=dict(color="#ef4444", width=0),
            name="Low", fill="tonexty",
            fillcolor="rgba(59,130,246,0.3)",
        ))
        fig.add_trace(go.Scatter(
            x=bb_data["date"], y=bb_data["bb_high"],
            mode="lines", line=dict(color="#10b981", width=2), name="Peak",
        ))
        fig.add_trace(go.Scatter(
            x=bb_data["date"], y=bb_data["bb_low"],
            mode="lines", line=dict(color="#ef4444", width=1.5, dash="dot"),
            name="Low point",
        ))
        fig.update_layout(margin=dict(t=5, b=0), height=300,
                          xaxis_title="", yaxis_title="Body battery",
                          yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green line = daily peak (after overnight recovery). Red = daily low (end of day). "
                   "Shaded band = your daily battery swing.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Overnight recovery (points gained)")
        if not bb_data.empty:
            fig = px.bar(bb_data, x="date", y="bb_recovery",
                         color="bb_recovery",
                         color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
                         range_color=[0, 80],
                         labels={"bb_recovery": "Points recovered", "date": ""})
            fig.update_layout(margin=dict(t=5, b=0), height=260,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            avg_rec = bb_data["bb_recovery"].mean()
            st.caption(f"Average overnight recovery: **{avg_rec:.0f} points**. "
                       "Healthy range is 40–70+.")

    with col_b:
        st.subheader("Daily stress level")
        stress_data = well_df.dropna(subset=["stress_avg"])
        if not stress_data.empty:
            stress_data = stress_data.copy()
            stress_data["rolling_stress"] = stress_data["stress_avg"].rolling(7, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=stress_data["date"], y=stress_data["stress_avg"],
                                 marker_color="#6366f1", opacity=0.4, name="Daily"))
            fig.add_trace(go.Scatter(x=stress_data["date"],
                                     y=stress_data["rolling_stress"],
                                     line=dict(color="#f59e0b", width=2),
                                     name="7-day avg"))
            fig.add_hline(y=50, line_dash="dash", line_color="red",
                          annotation_text="High stress threshold")
            fig.update_layout(margin=dict(t=5, b=0), height=260,
                              xaxis_title="", yaxis_title="Stress (0-100)")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily steps")
    steps_data = well_df.dropna(subset=["steps"])
    if not steps_data.empty:
        steps_data = steps_data.copy()
        steps_data["rolling_steps"] = steps_data["steps"].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=steps_data["date"], y=steps_data["steps"],
                             marker_color="#8b5cf6", opacity=0.5, name="Daily steps"))
        fig.add_trace(go.Scatter(x=steps_data["date"], y=steps_data["rolling_steps"],
                                 line=dict(color="#f59e0b", width=2), name="7-day avg"))
        fig.add_hline(y=10000, line_dash="dash", line_color="#10b981",
                      annotation_text="10k goal")
        fig.update_layout(margin=dict(t=5, b=0), height=260,
                          xaxis_title="", yaxis_title="Steps")
        st.plotly_chart(fig, use_container_width=True)

    # Body battery vs run performance correlation
    st.subheader("Body battery → next run performance")
    if not bb_data.empty and not runs_df.empty:
        run_dates = runs_df.copy()
        run_dates["prev_bb"] = run_dates["date"].apply(
            lambda d: bb_data[bb_data["date"] < d]["bb_high"].iloc[-1]
            if not bb_data[bb_data["date"] < d].empty else None
        )
        corr_df = run_dates.dropna(subset=["prev_bb", "pace_min_km"])
        corr_df = corr_df[corr_df["pace_min_km"] < 10]
        if len(corr_df) > 5:
            fig = px.scatter(corr_df, x="prev_bb", y="pace_min_km",
                             trendline="ols",
                             color_discrete_sequence=["#3b82f6"],
                             labels={"prev_bb": "Body battery before run",
                                     "pace_min_km": "Pace (min/km)"},
                             hover_data={"date": True, "distance_km": True})
            fig.update_layout(margin=dict(t=5, b=0), height=280,
                              yaxis_autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("If the trendline slopes down-right: higher battery = faster pace. "
                       "Shows how recovery predicts performance.")


# ── TAB 4: Race Predictor ─────────────────────────────────────────────────────
with tab4:
    st.subheader("Half marathon predictor")
    st.markdown("Predictions use two methods: **Riegel formula** (from your actual runs) "
                "and **Jack Daniels VO2max** (from your fitness score).")

    long_runs = runs_df[runs_df["distance_km"] >= 8].sort_values("date", ascending=False)
    latest_vo2 = runs_df["vo2max"].dropna().iloc[-1] if runs_df["vo2max"].notna().any() else None

    col_a, col_b, col_c = st.columns(3)

    # Riegel prediction
    with col_a:
        st.markdown("#### Riegel prediction")
        if not long_runs.empty:
            best = long_runs.iloc[0]
            hm_min = riegel_predict(best["duration_min"], best["distance_km"])
            st.metric("Predicted HM time", seconds_to_hms(hm_min * 60))
            st.caption(f"Based on {best['distance_km']:.1f} km run on "
                       f"{best['date'].strftime('%d %b')} "
                       f"({pace_str(best['pace_min_km'])})")
        else:
            st.info("Need a run ≥ 8 km for Riegel prediction.")

    # Daniels VO2max prediction
    with col_b:
        st.markdown("#### VO2max prediction")
        if latest_vo2:
            hm_min_v = vo2max_to_hm_prediction(latest_vo2)
            st.metric("Predicted HM time", seconds_to_hms(hm_min_v * 60))
            st.caption(f"Based on VO2max {latest_vo2:.0f} — Jack Daniels VDOT method")
        else:
            st.info("No VO2max data available.")

    # Consensus
    with col_c:
        st.markdown("#### Race pace")
        predictions = []
        if not long_runs.empty:
            predictions.append(riegel_predict(long_runs.iloc[0]["duration_min"],
                                              long_runs.iloc[0]["distance_km"]))
        if latest_vo2:
            predictions.append(vo2max_to_hm_prediction(latest_vo2))
        if predictions:
            avg_min = sum(predictions) / len(predictions)
            pace_hm = avg_min / 21.0975
            st.metric("Target race pace", pace_str(pace_hm))
            st.caption("Average of available predictions")

    # Training zones
    if latest_vo2:
        st.divider()
        st.subheader("Jack Daniels training paces")
        paces = training_paces(latest_vo2)
        cols = st.columns(len(paces))
        descriptions = {
            "Easy": "Recovery & long runs — conversational",
            "Marathon": "Comfortably hard — 3-4 hr effort",
            "Threshold": "Comfortably hard — 20-40 min sustained",
            "Interval": "Hard — 3-5 min reps at VO2max",
        }
        for i, (zone, pace) in enumerate(paces.items()):
            with cols[i]:
                st.metric(zone, pace_str(pace))
                st.caption(descriptions.get(zone, ""))

    # Prediction over time
    st.divider()
    st.subheader("Predicted HM time — trend over 6 months")
    vo2_series = runs_df.dropna(subset=["vo2max"]).copy()
    if not vo2_series.empty:
        vo2_series["predicted_hm_min"] = vo2_series["vo2max"].apply(vo2max_to_hm_prediction)
        vo2_series["predicted_hm_str"] = vo2_series["predicted_hm_min"].apply(
            lambda m: seconds_to_hms(m * 60))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vo2_series["date"], y=vo2_series["predicted_hm_min"],
            mode="lines+markers", line=dict(color="#f59e0b", width=2),
            marker=dict(size=8),
            hovertemplate="<b>%{text}</b><br>%{x|%d %b}<extra></extra>",
            text=vo2_series["predicted_hm_str"],
        ))
        fig.update_layout(margin=dict(t=5, b=0), height=260,
                          xaxis_title="", yaxis_title="Predicted time (min)",
                          yaxis_autorange="reversed")
        ymin = vo2_series["predicted_hm_min"].min()
        ymax = vo2_series["predicted_hm_min"].max()
        fig.update_yaxes(
            tickvals=list(range(int(ymin) - 5, int(ymax) + 10, 5)),
            ticktext=[seconds_to_hms(v * 60) for v in range(int(ymin) - 5, int(ymax) + 10, 5)],
        )
        st.plotly_chart(fig, use_container_width=True)
        delta = vo2_series["predicted_hm_min"].iloc[0] - vo2_series["predicted_hm_min"].iloc[-1]
        if delta > 0:
            st.caption(f"Your predicted HM time has improved by **{seconds_to_hms(abs(delta)*60)}** over the last 6 months. 📈")
        else:
            st.caption("Predicted HM time is flat or slower over 6 months.")

    # AI race analysis
    st.divider()
    if st.button("Get AI race readiness assessment", type="primary"):
        with st.spinner("Analysing race readiness..."):
            analysis = auto_analysis(act_df, well_df, focus="race")
        st.markdown(analysis)


# ── TAB 5: AI Coach ───────────────────────────────────────────────────────────
with tab5:
    st.subheader("Training Analysis")
    st.markdown("Analyses your full 6-month training history and gives you a personalised coaching assessment.")

    if st.button("Generate analysis", type="primary"):
        result = auto_analysis(act_df, well_df, focus="general")
        st.markdown(result)
