#!/usr/bin/env python3
"""Sync Garmin Connect data to local markdown files or a remote endpoint.

Built on the python-garminconnect library by cyberjunky:
https://github.com/cyberjunky/python-garminconnect
"""

import argparse
import base64
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from garminconnect import Garmin

TOKENSTORE_DIR = Path.home() / ".garminconnect"


def get_client(token_b64: str | None = None) -> Garmin:
    """Return an authenticated Garmin client."""
    garmin = Garmin()

    if token_b64:
        token_data = base64.b64decode(token_b64).decode()
        garmin.login(tokenstore=token_data)
    else:
        tokenstore_path = str(TOKENSTORE_DIR)
        if TOKENSTORE_DIR.exists():
            garmin.login(tokenstore=tokenstore_path)
        else:
            raise SystemExit(
                "No saved tokens found. Run with --login first, or set GARMIN_TOKEN_B64."
            )
    return garmin


def do_login() -> str:
    """Interactive login with email/password. Returns base64 token bundle."""
    email = os.environ.get("GARMIN_EMAIL")
    password = os.environ.get("GARMIN_PASSWORD")
    if not email or not password:
        raise SystemExit("Set GARMIN_EMAIL and GARMIN_PASSWORD environment variables.")

    garmin = Garmin(email=email, password=password, prompt_mfa=mfa_prompt)
    tokenstore_path = str(TOKENSTORE_DIR)
    garmin.login(tokenstore=tokenstore_path)

    token_data = garmin.client.dumps()
    token_b64 = base64.b64encode(token_data.encode()).decode()

    print("Login successful. Tokens saved to", tokenstore_path)
    print()
    print("=== BASE64 TOKEN BUNDLE (copy this for GitHub Actions) ===")
    print(token_b64)
    print("=== END TOKEN BUNDLE ===")
    return token_b64


def mfa_prompt() -> str:
    return input("Enter MFA/2FA code: ").strip()


def fetch_wellness(garmin: Garmin, day: date) -> dict:
    """Fetch all wellness metrics for a single day."""
    day_str = day.isoformat()
    wellness = {}

    try:
        summary = garmin.get_user_summary(day_str)
        wellness["steps"] = summary.get("totalSteps")
        wellness["calories"] = summary.get("totalKilocalories")
        wellness["distance_m"] = summary.get("totalDistanceMeters")
    except Exception:
        pass

    try:
        hr = garmin.get_resting_heart_rate(day_str)
        rhr_data = hr.get("restingHeartRate") if isinstance(hr, dict) else None
        if isinstance(rhr_data, dict):
            wellness["resting_hr"] = rhr_data.get("value")
        elif isinstance(rhr_data, (int, float)):
            wellness["resting_hr"] = rhr_data
    except Exception:
        pass

    try:
        hrv = garmin.get_hrv_data(day_str)
        if isinstance(hrv, dict):
            summary_val = hrv.get("hrvSummary") or hrv.get("summary") or hrv
            if isinstance(summary_val, dict):
                wellness["hrv"] = (
                    summary_val.get("lastNightAvg")
                    or summary_val.get("weeklyAvg")
                    or summary_val.get("lastNight")
                )
    except Exception:
        pass

    try:
        sleep = garmin.get_sleep_data(day_str)
        if isinstance(sleep, dict):
            dur = sleep.get("dailySleepDTO", {})
            seconds = dur.get("sleepTimeInSeconds")
            if seconds:
                wellness["sleep_hours"] = round(seconds / 3600, 1)
            wellness["sleep_score"] = (
                sleep.get("overallScore", {}).get("value")
                if isinstance(sleep.get("overallScore"), dict)
                else sleep.get("sleepScores", {}).get("overall")
            )
    except Exception:
        pass

    try:
        end_day = day + timedelta(days=1)
        bb = garmin.get_body_battery(day_str, end_day.isoformat())
        if isinstance(bb, list) and bb:
            charged = [p for p in bb if isinstance(p, dict) and p.get("charged") is not None]
            if charged:
                vals = [p.get("charged", 0) for p in charged]
                wellness["body_battery_low"] = min(vals) if vals else None
                wellness["body_battery_high"] = max(vals) if vals else None
        elif isinstance(bb, dict):
            items = bb.get("bodyBatteryValuesArray") or bb.get("dateTimeBBList") or []
            if items:
                nums = [v[-1] if isinstance(v, list) else v.get("value", 0) for v in items if v]
                nums = [n for n in nums if isinstance(n, (int, float))]
                if nums:
                    wellness["body_battery_low"] = min(nums)
                    wellness["body_battery_high"] = max(nums)
    except Exception:
        pass

    try:
        stress = garmin.get_all_day_stress(day_str)
        if isinstance(stress, dict):
            wellness["stress_avg"] = stress.get("overallStressLevel") or stress.get("avgStressLevel")
        elif isinstance(stress, list) and stress:
            vals = [s.get("stressLevel", 0) for s in stress if isinstance(s, dict) and s.get("stressLevel")]
            wellness["stress_avg"] = round(sum(vals) / len(vals)) if vals else None
    except Exception:
        pass

    try:
        tr = garmin.get_training_readiness(day_str)
        if isinstance(tr, dict):
            wellness["training_readiness"] = (
                tr.get("score") or tr.get("trainingReadinessScore") or tr.get("value")
            )
        elif isinstance(tr, list) and tr:
            wellness["training_readiness"] = tr[0].get("score") if isinstance(tr[0], dict) else None
    except Exception:
        pass

    wellness["date"] = day_str
    return wellness


def fetch_activities(garmin: Garmin, start: date, end: date) -> list[dict]:
    """Fetch activities in a date range."""
    raw = garmin.get_activities_by_date(start.isoformat(), end.isoformat())
    if not isinstance(raw, list):
        return []

    activities = []
    for a in raw:
        act = {
            "date": a.get("startTimeLocal", "")[:10],
            "name": a.get("activityName", "Activity"),
            "type": a.get("activityType", {}).get("typeKey", "unknown")
                   if isinstance(a.get("activityType"), dict)
                   else a.get("activityType", "unknown"),
            "duration_s": a.get("duration"),
            "distance_m": a.get("distance"),
            "calories": a.get("calories"),
            "avg_hr": a.get("averageHR"),
            "max_hr": a.get("maxHR"),
            "avg_power": a.get("avgPower"),
            "training_effect_aerobic": a.get("aerobicTrainingEffect"),
            "training_effect_anaerobic": a.get("anaerobicTrainingEffect"),
            "vo2max": a.get("vO2MaxValue"),
        }
        activities.append(act)
    return activities


def format_duration(seconds: float | None) -> str:
    if not seconds:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def format_distance(meters: float | None) -> str:
    if not meters:
        return ""
    km = meters / 1000
    if km >= 1:
        return f"{km:.2f} km"
    return f"{int(meters)} m"


def wellness_to_markdown(w: dict) -> str:
    lines = [f"# Garmin wellness {w['date']}"]
    if w.get("resting_hr"):
        lines.append(f"- Resting HR: {w['resting_hr']} bpm")
    if w.get("hrv"):
        lines.append(f"- HRV (overnight): {w['hrv']} ms")
    if w.get("sleep_hours"):
        score_part = f" (score {w['sleep_score']})" if w.get("sleep_score") else ""
        lines.append(f"- Sleep: {w['sleep_hours']} h{score_part}")
    if w.get("body_battery_low") is not None and w.get("body_battery_high") is not None:
        lines.append(f"- Body battery: {w['body_battery_low']} -> {w['body_battery_high']}")
    if w.get("stress_avg"):
        lines.append(f"- Stress (avg): {w['stress_avg']}")
    if w.get("steps"):
        lines.append(f"- Steps: {w['steps']}")
    if w.get("training_readiness"):
        lines.append(f"- Training readiness: {w['training_readiness']}")
    return "\n".join(lines) + "\n"


def activity_to_markdown(a: dict) -> str:
    lines = [f"# {a['name']}"]
    lines.append(f"- Type: {a['type']}")
    lines.append(f"- Date: {a['date']}")
    lines.append(f"- Duration: {format_duration(a.get('duration_s'))}")
    dist = format_distance(a.get("distance_m"))
    if dist:
        lines.append(f"- Distance: {dist}")
    if a.get("calories"):
        lines.append(f"- Calories: {a['calories']}")
    if a.get("avg_hr"):
        lines.append(f"- Avg HR: {a['avg_hr']} bpm")
    if a.get("max_hr"):
        lines.append(f"- Max HR: {a['max_hr']} bpm")
    if a.get("avg_power"):
        lines.append(f"- Avg power: {a['avg_power']} W")
    if a.get("training_effect_aerobic"):
        lines.append(f"- Aerobic TE: {a['training_effect_aerobic']}")
    if a.get("training_effect_anaerobic"):
        lines.append(f"- Anaerobic TE: {a['training_effect_anaerobic']}")
    if a.get("vo2max"):
        lines.append(f"- VO2max: {a['vo2max']}")
    return "\n".join(lines) + "\n"


def sink_files(activities: list[dict], wellness: list[dict], out_dir: Path):
    daily_dir = out_dir / "daily"
    act_dir = out_dir / "activities"
    daily_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    for w in wellness:
        path = daily_dir / f"{w['date']}.md"
        path.write_text(wellness_to_markdown(w), encoding="utf-8")

    for a in activities:
        slug = a.get("name", "activity").lower().replace(" ", "-")[:40]
        path = act_dir / f"{a['date']}-{slug}.md"
        path.write_text(activity_to_markdown(a), encoding="utf-8")

    data_path = out_dir / "data.json"
    store = {}
    if data_path.exists():
        try:
            store = json.loads(data_path.read_text(encoding="utf-8"))
        except Exception:
            store = {}

    for w in wellness:
        store.setdefault("wellness", {})[w["date"]] = w
    for a in activities:
        store.setdefault("activities", {}).setdefault(a["date"], []).append(a)

    data_path.write_text(json.dumps(store, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(wellness)} daily + {len(activities)} activity files to {out_dir}/")


def sink_supabase(activities: list[dict], wellness: list[dict]):
    import requests

    url = os.environ.get("GARMIN_INGEST_URL")
    secret = os.environ.get("GARMIN_INGEST_SECRET")
    if not url:
        raise SystemExit("Set GARMIN_INGEST_URL for supabase sink.")

    payload = {"activities": activities, "wellness": wellness}
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["Authorization"] = f"Bearer {secret}"

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    print(f"Posted {len(activities)} activities + {len(wellness)} wellness records -> {resp.status_code}")


def main():
    parser = argparse.ArgumentParser(description="Sync Garmin Connect data")
    parser.add_argument("--login", action="store_true", help="Log in and save tokens")
    parser.add_argument("--days", type=int, default=3, help="Number of days to fetch (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Print data without writing")
    parser.add_argument("--sink", choices=["files", "supabase"], default="files", help="Output mode")
    parser.add_argument("--out", type=str, default="garmin", help="Output directory for files sink")
    args = parser.parse_args()

    if args.login:
        do_login()
        return

    token_b64 = os.environ.get("GARMIN_TOKEN_B64")
    garmin = get_client(token_b64)

    end = date.today()
    start = end - timedelta(days=args.days)

    print(f"Fetching {args.days} days: {start} to {end}")

    wellness_days = []
    for i in range(args.days):
        day = start + timedelta(days=i + 1)
        w = fetch_wellness(garmin, day)
        wellness_days.append(w)

    activities = fetch_activities(garmin, start, end)

    if args.dry_run:
        print("\n=== WELLNESS ===")
        for w in wellness_days:
            print(wellness_to_markdown(w))
        print("=== ACTIVITIES ===")
        for a in activities:
            print(activity_to_markdown(a))
        return

    if args.sink == "files":
        sink_files(activities, wellness_days, Path(args.out))
    elif args.sink == "supabase":
        sink_supabase(activities, wellness_days)


if __name__ == "__main__":
    main()
