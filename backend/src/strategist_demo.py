import os
from pathlib import Path
from typing import Any, Dict, Optional

# Local analysis function
from analyzer import analyze_match, insights_to_markdown


def run_local_strategist(json_path: str) -> str:
    ins = analyze_match(Path(json_path))
    md = insights_to_markdown(ins)
    return md


def try_azure_strategist(json_path: str) -> Optional[str]:
    """Use Azure OpenAI (gpt-4.1) to refine/expand the local Markdown insights.
    Returns final Markdown or None on failure.
    """
    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception:
        return None

    endpoint = "https://gpt-4o-intern.openai.azure.com/"
    deployment = "gpt-4.1"
    api_version = "2024-12-01-preview"
    subscription_key = (
        "BmaiYil8P7o3Dgv0JzIEIA4JYd3AHl7Jh6SzBdjkwXfF4DNxCzC3JQQJ99BGACYeBjFXJ3w3AAABACOGZkhi"
    )

    # Base MD from local analysis (avoid passing raw large JSON)
    base_md = run_local_strategist(json_path)

    ins = analyze_match(Path(json_path))
    meta_lines = []
    if ins.match.teams:
        meta_lines.append(f"Teams: {ins.match.teams[0]} vs {ins.match.teams[1]}")
    if ins.match.match_type:
        meta_lines.append(f"Match Type: {ins.match.match_type}")
    if ins.match.winner:
        by_txt = (
            f" by {next(iter(ins.match.result_by))} {next(iter(ins.match.result_by.values()))}"
            if ins.match.result_by else ""
        )
        meta_lines.append(f"Result: {ins.match.winner}{by_txt}")

    system_msg = (
        "You are the Tactical & Simulation Agent (Strategist). Refine and expand a Markdown insights report. "
        "Keep the existing structure and headings; improve Tactical Notes with concise, Test-specific, actionable items. "
        "Do not invent unavailable statistics; acknowledge data truncation. Return only the final Markdown."
    )
    user_msg = (
        "Base Insights Markdown to refine:\n\n" + base_md +
        "\n\nContext Metadata (for reference):\n" + "\n".join(f"- {l}" for l in meta_lines)
    )

    try:
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=2048,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment,
        )
        content = response.choices[0].message.content if response and response.choices else None
        if content and isinstance(content, str):
            return content
    except Exception:
        return None
    return None


def try_autogen_strategist(json_path: str) -> str:
    """
    A minimal AutoGen setup with two agents:
    - Strategist (assistant)
    - Analyst (assistant) delegated to call the local analyzer
    If OPENAI_API_KEY (or AOAI equivalents) is not set, we fallback to local.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return run_local_strategist(json_path)

    try:
        # Lazy import; keep demo runnable without this package. If not installed,
        # we fall back to local analysis seamlessly.
        from autogen import ConversableAgent, GroupChat, GroupChatManager  # type: ignore
    except Exception:
        return run_local_strategist(json_path)

    system_strategist = (
        "You are the Tactical & Simulation Agent (Strategist). "
        "You reason about match context and produce concise, actionable insights."
    )
    system_analyst = (
        "You are a Data Analyst Agent. When asked for analysis, call the provided Python function to parse the JSON and return Markdown."
    )

    strategist = ConversableAgent(
        name="Strategist",
        system_message=system_strategist,
        llm_config={
            "config_list": [
                {
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "api_key": api_key,
                }
            ],
            "temperature": 0.2,
        },
        human_input_mode="NEVER",
    )

    analyst = ConversableAgent(
        name="Analyst",
        system_message=system_analyst,
        llm_config={
            "config_list": [
                {
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "api_key": api_key,
                }
            ],
            "temperature": 0.1,
        },
        human_input_mode="NEVER",
    )

    # Tool: local function for analysis
    def analyze_tool(message: Dict[str, Any]):
        return run_local_strategist(json_path)

    analyst.register_reply(
        triggers={"role": "user"},
        reply_func=lambda sender, recipient, context: (
            True,
            analyze_tool(context.get("message", {})),
        ),
    )

    groupchat = GroupChat(agents=[strategist, analyst], messages=[], max_round=2)
    manager = GroupChatManager(groupchat=groupchat, llm_config=strategist.llm_config)

    prompt = (
        "Analyze the provided match JSON path and deliver a Markdown insights report: "
        f"{json_path}. If analysis is available via tools, use it. Keep sections: Summary, Tactical Notes, Data Completeness."
    )
    res = strategist.initiate_chat(manager, message=prompt)

    # Extract final content
    final = res.chat_history[-1]["content"] if res and getattr(res, "chat_history", None) else None
    return final or run_local_strategist(json_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Strategist Agent Demo")
    ap.add_argument("json", type=str, help="Path to match JSON (e.g., 1244025.json)")
    ap.add_argument("--out", type=str, default=None, help="Output Markdown path")
    args = ap.parse_args()

    # Preference order: Azure OpenAI -> AutoGen -> Local
    md = try_azure_strategist(args.json) or try_autogen_strategist(args.json) or run_local_strategist(args.json)
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "insights" / (Path(args.json).stem + "_insights.md")
    out.write_text(md, encoding="utf-8")
    print(f"Wrote insights to {out}")
