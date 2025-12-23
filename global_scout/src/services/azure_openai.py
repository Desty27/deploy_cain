from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, cast

from openai import AzureOpenAI

AZURE_DEFAULT_DEPLOYMENT = "gpt-4.1"
AZURE_DEFAULT_API_VERSION = "2024-12-01-preview"


def _build_client() -> Optional[tuple[AzureOpenAI, str]]:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", AZURE_DEFAULT_DEPLOYMENT)
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", AZURE_DEFAULT_API_VERSION)

    if not api_key or not endpoint or not deployment:
        return None

    try:
        client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
    except Exception:
        return None
    return client, deployment


def generate_medical_coordination_notes(
    match_context: Dict[str, str],
    roster_snapshot: Dict[str, object],
    base_notes: List[str],
    temperature: float = 0.3,
) -> Optional[List[str]]:
    client_info = _build_client()
    if client_info is None:
        return None

    client, deployment = client_info

    team_a = match_context.get("team_a", "Team A")
    team_b = match_context.get("team_b", "Team B")
    venue = match_context.get("venue", "unknown venue")
    date = match_context.get("date", "TBD")

    high_risk_text = roster_snapshot.get("high_risk_report", "None")
    moderate_risk_text = roster_snapshot.get("moderate_risk_report", "None")
    readiness_mean = roster_snapshot.get("readiness_mean", 0)
    workload_trends = roster_snapshot.get("workload_flags", "None")
    per_team_high = cast(Dict[str, str], roster_snapshot.get("per_team_high") or {})
    per_team_moderate = cast(Dict[str, str], roster_snapshot.get("per_team_moderate") or {})
    per_team_workload = cast(Dict[str, str], roster_snapshot.get("per_team_workload") or {})

    # Build a match-specific prompt including per-team breakdowns where available
    team_breakdown_lines = []
    for t in (team_a, team_b):
        th = per_team_high.get(t, "None")
        tm = per_team_moderate.get(t, "None")
        tw = per_team_workload.get(t, "None")
        team_breakdown_lines.append(f"{t}: High-risk -> {th}; Moderate-risk -> {tm}; Workload -> {tw}.")

    user_prompt = (
        "You are the Digital Physio agent collaborating with the Tactical agent in a cricket analyst war-room. "
        "Produce 3 match-specific recommendations as JSON (no markdown).\n\n"
        f"Match: {team_a} vs {team_b} on {date} at {venue}.\n"
        f"Squad readiness avg: {readiness_mean:.2f}.\n"
        f"Global high-risk players: {high_risk_text}.\n"
        f"Global moderate-risk players: {moderate_risk_text}.\n"
        f"Global workload considerations: {workload_trends}.\n"
        "Team breakdown:\n"
        + "\n".join(team_breakdown_lines)
        + "\n\n"
        "Respond with a JSON array of exactly 3 objects. Each object must contain keys:"
        " team (string), focus (string describing risk insight), medical_action (string), tactical_adjustment (string)."
        "Keep language concise (<30 words per field) and reference specific players/roles."
        "Example: [{\"team\":\"England\",\"focus\":\"...\",\"medical_action\":\"...\",\"tactical_adjustment\":\"...\"}, ...]."
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are CAIN's Digital Physio expert. Provide medically grounded, tactically actionable guidance. "
                        "Return only valid JSON (no Markdown, no code fences)."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=350,
            top_p=0.9,
        )
    except Exception:
        return None

    text = response.choices[0].message.content if response.choices else ""
    if not text:
        return None

    bullets: List[str] = []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            for item in payload[:5]:
                if isinstance(item, dict):
                    team = str(item.get("team", "Team"))
                    focus = str(item.get("focus", "Key focus"))
                    medical = str(item.get("medical_action", "Medical action"))
                    tactical = str(item.get("tactical_adjustment", "Tactical adjustment"))
                    bullets.append(
                        f"[{team}] {focus} — Medical: {medical}. Tactical: {tactical}."
                    )
    except json.JSONDecodeError:
        bullets = []

    if bullets:
        return bullets

    # Fallback: split plain text into lines, merge with base notes for safety
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines
    return base_notes or None


def generate_integrity_brief(
    match_meta: Dict[str, Any],
    pressure_windows: List[Dict[str, Any]],
    hotspots: List[Dict[str, Any]],
    alerts: List[str],
    temperature: float = 0.2,
) -> Optional[str]:
    client_info = _build_client()
    if client_info is None:
        return None

    client, deployment = client_info

    team_a, team_b = (match_meta.get("teams") or ["Team A", "Team B"])[:2]
    venue = match_meta.get("venue", "Unknown venue")
    date = match_meta.get("date", "TBD")

    user_prompt = (
        "You are CAIN's Adjudication & Integrity Agent. Compose a concise integrity brief as Markdown.\n"
        "Context: {team_a} vs {team_b} on {date} at {venue}.\n"
        "Appeal pressure windows (JSON): {pressure}\n"
        "Review hotspots (JSON): {hotspots}\n"
        "Alerts: {alerts}\n"
        "Respond with headings: Situation Snapshot, High-Risk Windows, Coordination Signals."
        "Limit total length to ~180 words. Highlight coordination cues for Tactical and Physio agents where relevant."
    ).format(
        team_a=team_a,
        team_b=team_b,
        date=date,
        venue=venue,
        pressure=json.dumps(pressure_windows),
        hotspots=json.dumps(hotspots),
        alerts="; ".join(alerts),
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the Integrity co-pilot for CAIN. Provide objective, data-backed umpiring support."
                        " Keep analysis neutral and action-oriented."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=350,
            top_p=0.9,
        )
    except Exception:
        return None

    text = response.choices[0].message.content if response.choices else ""
    return text or None


def generate_supervisor_plan(
    match_meta: Dict[str, Any],
    tactical_snapshot: Dict[str, Any],
    wellness_snapshot: Dict[str, Any],
    integrity_snapshot: Dict[str, Any],
    recruiting_snapshot: Dict[str, Any],
    temperature: float = 0.2,
) -> Optional[str]:
    client_info = _build_client()
    if client_info is None:
        return None

    client, deployment = client_info

    meta_text = json.dumps(match_meta, ensure_ascii=False)
    tactical_text = json.dumps(tactical_snapshot, ensure_ascii=False)
    wellness_text = json.dumps(wellness_snapshot, ensure_ascii=False)
    integrity_text = json.dumps(integrity_snapshot, ensure_ascii=False)
    recruiting_text = json.dumps(recruiting_snapshot, ensure_ascii=False)

    prompt = (
        "You are the CAIN Supervisor Agent (Head Coach)."  # noqa: E501
        " Produce a succinct Markdown mission briefing aligning Tactical, Physio, Integrity, and Scout agents.\n"
        "Context JSON blocks are provided; synthesize them into no more than 180 words.\n"
        "Structure the briefing with the headings: Goal Alignment, Tactical Priorities, Human Performance, Integrity Watch, Recruitment Lens.\n"
        "Each section should contain 2 bullet points max, referencing specific data cues from the context."  # noqa: E501
    )

    user_payload = (
        f"MATCH_META: {meta_text}\n"
        f"TACTICAL: {tactical_text}\n"
        f"WELLNESS: {wellness_text}\n"
        f"INTEGRITY: {integrity_text}\n"
        f"RECRUITING: {recruiting_text}"
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You orchestrate CAIN's multi-agent collaboration. Prioritise clarity, brevity, and actionable sequencing."  # noqa: E501
                    ),
                },
                {"role": "user", "content": prompt + "\n\n" + user_payload},
            ],
            temperature=temperature,
            max_tokens=380,
            top_p=0.9,
        )
    except Exception:
        return None

    text = response.choices[0].message.content if response.choices else ""
    return text or None
