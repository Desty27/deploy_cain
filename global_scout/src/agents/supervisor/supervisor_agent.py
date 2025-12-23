from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from global_scout.src.services.azure_openai import generate_supervisor_plan


@dataclass
class SupervisorOutputs:
    plan_markdown: Optional[str]
    tactical_callouts: List[str]
    wellness_callouts: List[str]
    integrity_callouts: List[str]
    recruiting_prompts: List[str]
    coordination_queue: List[str]


class SupervisorAgent:
    """Supervisor orchestrator synthesising multi-agent signals into a unified plan."""

    def __init__(
        self,
        match_meta: Dict[str, Any],
        tactical_snapshot: Dict[str, Any],
        wellness_snapshot: Optional[Dict[str, Any]] = None,
        integrity_snapshot: Optional[Dict[str, Any]] = None,
        recruiting_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.match_meta = match_meta
        self.tactical_snapshot = tactical_snapshot
        self.wellness_snapshot = wellness_snapshot or {}
        self.integrity_snapshot = integrity_snapshot or {}
        self.recruiting_snapshot = recruiting_snapshot or {}

    def _derive_tactical_notes(self) -> List[str]:
        notes: List[str] = []
        phase = self.tactical_snapshot.get("phase_team", {})
        for team, phases in list(phase.items())[:2]:
            worst_phase = None
            lowest_rpo = 999.0
            for pname, stats in (phases or {}).items():
                rpo = stats.get("rpo", 0)
                if stats.get("balls", 0) and rpo < lowest_rpo:
                    lowest_rpo = rpo
                    worst_phase = pname
            if worst_phase is not None:
                notes.append(f"{team}: bolster {worst_phase} scoring (RPO {lowest_rpo:.2f}).")
        wicket_clusters = self.tactical_snapshot.get("wicket_clusters", [])
        if wicket_clusters:
            top = wicket_clusters[0]
            notes.append(
                f"Exploit wicket surge: innings {top['innings']} overs {top['start_over']}-{top['end_over']} ({top['wickets']} wickets)."
            )
        bowler_clusters = self.tactical_snapshot.get("bowler_clusters", [])
        if bowler_clusters:
            bow = bowler_clusters[0]
            notes.append(
                f"Preserve {bow.get('bowler_name', 'key bowler')} for overs {bow['start_over']}-{bow['end_over']} (innings {bow['innings']})."
            )
        if not notes:
            notes.append("Stabilise phase tempo; deploy Tactical Agent for detailed drill scripts.")
        return notes

    def _derive_wellness_notes(self) -> List[str]:
        notes: List[str] = []
        if not self.wellness_snapshot:
            return notes
        high = self.wellness_snapshot.get("high_risk_names") or []
        if high:
            notes.append("High-risk athletes: " + ", ".join(high[:3]))
        readiness = self.wellness_snapshot.get("readiness_mean")
        if readiness is not None:
            notes.append(f"Squad readiness mean {readiness:.2f}.")
        driver = self.wellness_snapshot.get("top_driver")
        if driver:
            notes.append(f"Primary strain driver: {driver}.")
        if not notes:
            notes.append("Wellness steady; maintain recovery cadence.")
        return notes

    def _derive_integrity_notes(self) -> List[str]:
        alerts = self.integrity_snapshot.get("alerts") or []
        if alerts:
            return alerts[:4]
        return ["No integrity anomalies detected; standard officiating support."]

    def _derive_recruiting_prompts(self) -> List[str]:
        prompts: List[str] = []
        shortlist = self.recruiting_snapshot.get("shortlist_highlights") or []
        if shortlist:
            prompts.append("Scout focus: " + "; ".join(shortlist[:3]))
        else:
            prompts.append("Direct Global Scout to refresh shortlist aligned with match objectives.")
        return prompts

    def _derive_coordination(self, tactical: List[str], wellness: List[str], integrity: List[str]) -> List[str]:
        queue: List[str] = []
        if tactical:
            queue.append(f"Synchronise Tactical Agent on: {tactical[0]}")
        if wellness:
            queue.append(f"Loop Physio Agent for load plan: {wellness[0]}")
        if integrity:
            queue.append(f"Brief Integrity Agent: {integrity[0]}")
        return queue

    def build(self) -> SupervisorOutputs:
        tactical_notes = self._derive_tactical_notes()
        wellness_notes = self._derive_wellness_notes()
        integrity_notes = self._derive_integrity_notes()
        recruiting_prompts = self._derive_recruiting_prompts()
        coordination = self._derive_coordination(tactical_notes, wellness_notes, integrity_notes)

        plan_markdown = generate_supervisor_plan(
            match_meta=self.match_meta,
            tactical_snapshot=self.tactical_snapshot,
            wellness_snapshot=self.wellness_snapshot,
            integrity_snapshot=self.integrity_snapshot,
            recruiting_snapshot=self.recruiting_snapshot,
        )

        return SupervisorOutputs(
            plan_markdown=plan_markdown,
            tactical_callouts=tactical_notes,
            wellness_callouts=wellness_notes,
            integrity_callouts=integrity_notes,
            recruiting_prompts=recruiting_prompts,
            coordination_queue=coordination,
        )


def build_agent(
    match_meta: Dict[str, Any],
    tactical_snapshot: Dict[str, Any],
    wellness_snapshot: Optional[Dict[str, Any]] = None,
    integrity_snapshot: Optional[Dict[str, Any]] = None,
    recruiting_snapshot: Optional[Dict[str, Any]] = None,
) -> SupervisorAgent:
    return SupervisorAgent(
        match_meta=match_meta,
        tactical_snapshot=tactical_snapshot,
        wellness_snapshot=wellness_snapshot,
        integrity_snapshot=integrity_snapshot,
        recruiting_snapshot=recruiting_snapshot,
    )
