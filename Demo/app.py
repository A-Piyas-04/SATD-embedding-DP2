"""Gradio demo for SATD identification + categorization.

Presentation-focused UI for live demos. Inference matches V2 notebook.
Adds: confidence bars, Title/Desc/Both compare, keyword cues.
"""

from __future__ import annotations

import traceback

import gradio as gr
import pandas as pd

from satd_infer import (
    SatdClassifier,
    default_model_dir,
    find_keyword_cues,
    highlight_keyword_cues,
)

# Short code → full name (shown on Final label)
CATEGORY_FULL = {
    "C/D": "Code / Design debt",
    "DOC": "Documentation debt",
    "REQ": "Requirement debt",
    "TES": "Test debt",
}

IDENT_FULL = {
    "SATD": "Self-Admitted Technical Debt",
    "Not-SATD": "Not technical debt",
}

EXAMPLES = [
    [
        "Add dark mode toggle to settings page",
        "Users asked for a dark theme. Implement a simple on/off switch in Settings.",
        "Title + Description",
    ],
    [
        "FIXME: temporary hack in payment parser",
        "We hardcoded a null check to stop crashes. This is ugly — refactor the parser properly later.",
        "Title + Description",
    ],
    [
        "Missing unit tests for login flow",
        "Authentication has almost no automated tests. We need unit tests for success, failure, and timeout cases.",
        "Title + Description",
    ],
    [
        "API docs are outdated for v2 endpoints",
        "The developer guide still describes v1 only. Documentation must be updated with examples for the new webhooks.",
        "Title + Description",
    ],
    [
        "Requirement never implemented: export to CSV",
        "The original requirement said users can export reports to CSV. That feature was promised but never built.",
        "Title + Description",
    ],
    [
        "Workaround: skip validation to ship tonight",
        "Leaving input validation disabled as a shortcut so we can release. Design debt — clean this up next sprint.",
        "Description only",
    ],
    [
        "Flaky tests disabled in CI",
        "Three integration tests fail randomly so we commented them out. We still need proper, stable test coverage.",
        "Title + Description",
    ],
    [
        "README does not explain how to run locally",
        "New contributors cannot start the project. Please write clear setup documentation in the README.",
        "Title + Description",
    ],
    [
        "Incomplete requirement: role-based access",
        "Spec required admin vs user permissions. Only admin exists today; the rest of the requirement is unfinished.",
        "Title + Description",
    ],
    [
        "Fix typo in welcome email subject line",
        "Change 'Welcom' to 'Welcome' in the email template. No other changes needed.",
        "Title + Description",
    ],
]

# Short labels for the always-visible right sidebar
SAMPLE_BUTTONS = [
    "1 · Feature request → Not-SATD",
    "2 · Hack / ugly code → C/D",
    "3 · Missing tests → TES",
    "4 · Outdated docs → DOC",
    "5 · Unbuilt requirement → REQ",
    "6 · Shortcut / design debt → C/D",
    "7 · Flaky tests → TES",
    "8 · Missing README → DOC",
    "9 · Incomplete requirement → REQ",
    "10 · Typo fix → Not-SATD",
]

EXAMPLE_HINTS = """
**What a human would expect (for the presenter):**

| # | Sample focus | Easy human answer |
|---|---|---|
| 1 | Normal feature request | **Not-SATD** |
| 2 | Hack / ugly code | **SATD → C/D** (Code / Design) |
| 3 | Missing tests | **SATD → TES** (Test) |
| 4 | Outdated docs | **SATD → DOC** (Documentation) |
| 5 | Unbuilt requirement | **SATD → REQ** (Requirement) |
| 6 | Shortcut / design debt | **SATD → C/D** |
| 7 | Disabled flaky tests | **SATD → TES** |
| 8 | Missing README docs | **SATD → DOC** |
| 9 | Incomplete requirement | **SATD → REQ** |
| 10 | Simple typo fix | **Not-SATD** |
"""

EMPTY_ID_PROBS = pd.DataFrame({"label": ["Not-SATD", "SATD"], "probability": [0.0, 0.0]})
EMPTY_CAT_PROBS = pd.DataFrame(
    {"label": ["C/D", "DOC", "REQ", "TES"], "probability": [0.0, 0.0, 0.0, 0.0]}
)
EMPTY_COMPARE = pd.DataFrame(
    columns=["Mode", "Identification", "Category", "Final label", "P(SATD)"]
)

clf = SatdClassifier()
_load_error: str | None = None

CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; }
.hero-box { padding: 0.1rem 0 0.25rem 0; }
.result-card textarea {
  font-size: 1.1rem !important;
  font-weight: 600 !important;
}
.sample-sidebar {
  position: sticky !important;
  top: 0.75rem;
  align-self: flex-start;
  max-height: calc(100vh - 1.5rem);
  overflow-y: auto;
  border-left: 1px solid rgba(128,128,128,0.25);
  padding-left: 0.75rem !important;
  background: var(--block-background-fill, transparent);
}
.sample-sidebar button {
  justify-content: flex-start !important;
  text-align: left !important;
  white-space: normal !important;
  height: auto !important;
  min-height: 2.4rem;
  padding: 0.45rem 0.65rem !important;
  font-size: 0.92rem !important;
}
"""


def _startup_status() -> str:
    global _load_error
    try:
        return clf.load()
    except Exception as exc:  # noqa: BLE001
        _load_error = f"{type(exc).__name__}: {exc}"
        return (
            f"Models not ready. {_load_error} "
            f"Put .joblib files in {default_model_dir()} (or set SATD_MODEL_DIR)."
        )


def format_identification(code: str) -> str:
    full = IDENT_FULL.get(code)
    return f"{code} — {full}" if full else code


def format_final_label(identification: str, category: str | None, final_label: str) -> str:
    if category:
        full = CATEGORY_FULL.get(category, "")
        return f"{category} — {full}" if full else category
    full = IDENT_FULL.get(final_label) or IDENT_FULL.get(identification)
    if full:
        return f"{final_label} — {full}"
    return final_label


def probs_to_df(probs: dict[str, float] | None, labels: list[str]) -> pd.DataFrame:
    probs = probs or {}
    return pd.DataFrame(
        {
            "label": labels,
            "probability": [round(float(probs.get(lab, 0.0)), 4) for lab in labels],
        }
    )


def _ensure_ready():
    if _load_error and not clf._loaded:
        raise RuntimeError(_load_error)
    if not clf._loaded:
        clf.load()


def classify(title: str, description: str, mode: str):
    try:
        _ensure_ready()
        result = clf.predict(title, description, mode)
    except Exception as exc:  # noqa: BLE001
        err = f"**Error:** {exc}\n\n```\n{traceback.format_exc()}\n```"
        return (
            err,
            "—",
            "—",
            "—",
            EMPTY_ID_PROBS,
            EMPTY_CAT_PROBS,
            "_Fix the error above, then try again._",
            "—",
        )

    under_budget = result.total_seconds < 60
    timing = (
        f"{result.total_seconds:.2f}s  ·  embed {result.embed_seconds:.2f}s  ·  "
        f"classify {result.classify_seconds:.3f}s  ·  {result.device}"
        + ("  ·  under 1 min ✓" if under_budget else "  ·  over 1 min")
    )

    preview = result.text_used
    if len(preview) > 360:
        preview = preview[:360] + "…"

    if result.category:
        step_story = (
            f"1. **Is it debt?** → {format_identification(result.identification)}\n"
            f"2. **What kind?** → **{result.category}** ({CATEGORY_FULL.get(result.category, '')})\n"
            f"3. **Final answer** → **{format_final_label(result.identification, result.category, result.final_label)}**"
        )
    else:
        step_story = (
            f"1. **Is it debt?** → {format_identification(result.identification)}\n"
            f"2. **What kind?** → skipped (not SATD)\n"
            f"3. **Final answer** → **{format_final_label(result.identification, None, result.final_label)}**"
        )

    detail = (
        f"### How the model decided\n{step_story}\n\n"
        f"**Text sent to the model** ({mode}):\n```\n{preview}\n```"
    )

    id_df = probs_to_df(result.identification_probs, ["Not-SATD", "SATD"])
    if result.category_probs:
        cat_df = probs_to_df(result.category_probs, ["C/D", "DOC", "REQ", "TES"])
    else:
        cat_df = EMPTY_CAT_PROBS.copy()

    cues_md = (
        f"**Cues spotted:** "
        + (
            ", ".join(f"`{c}`" for c in result.keyword_cues)
            if result.keyword_cues
            else "_No common SATD cue words found_"
        )
        + "\n\n"
        + highlight_keyword_cues(result.text_used)
    )

    return (
        detail,
        format_identification(result.identification),
        result.category if result.category else "—",
        format_final_label(result.identification, result.category, result.final_label),
        id_df,
        cat_df,
        cues_md,
        timing,
    )


def compare_modes(title: str, description: str):
    try:
        _ensure_ready()
        results, total = clf.predict_three_modes(title, description)
    except Exception as exc:  # noqa: BLE001
        return (
            EMPTY_COMPARE,
            f"**Error:** {exc}",
            f"{0:.2f}s",
        )

    rows = []
    for r in results:
        rows.append(
            {
                "Mode": r.mode,
                "Identification": r.identification,
                "Category": r.category or "—",
                "Final label": format_final_label(
                    r.identification, r.category, r.final_label
                ),
                "P(SATD)": round(float(r.identification_probs.get("SATD", 0.0)), 3),
            }
        )
    df = pd.DataFrame(rows)
    note = (
        "Same issue, three input variants (matches the applied Title / Description / Both experiment). "
        f"Total time **{total:.2f}s** on `{results[0].device}`."
    )
    timing = f"{total:.2f}s  ·  3 modes  ·  {results[0].device}"
    return df, note, timing


def refresh_cues(title: str, description: str, mode: str):
    text = SatdClassifier.build_text(clf, title, description, mode)
    cues = find_keyword_cues(text)
    cue_line = (
        ", ".join(f"`{c}`" for c in cues) if cues else "_No common SATD cue words found_"
    )
    return f"**Cues spotted:** {cue_line}\n\n{highlight_keyword_cues(text)}"


def load_sample(index: int):
    title, description, mode = EXAMPLES[index]
    return title, description, mode, refresh_cues(title, description, mode)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SATD Live Demo", css=CUSTOM_CSS) as demo:
        with gr.Column(elem_classes=["hero-box"]):
            gr.Markdown(
                """
                # Self-Admitted Technical Debt (SATD) Classifier
                Pick a sample on the **right**, then **Classify** or **Compare** on the left.
                """
            )

        with gr.Row(equal_height=False):
            # -------- Main workspace (left) --------
            with gr.Column(scale=7, min_width=520):
                with gr.Accordion("Label guide", open=False):
                    gr.Markdown(
                        """
                        | Code | Full name | Meaning |
                        |---|---|---|
                        | **Not-SATD** | Not technical debt | Normal work / bug / feature |
                        | **SATD** | Self-Admitted Technical Debt | Admits debt, hack, missing work |
                        | **C/D** | Code / Design debt | Ugly code, hack, workaround |
                        | **TES** | Test debt | Missing / skipped / weak tests |
                        | **REQ** | Requirement debt | Incomplete or never implemented |
                        | **DOC** | Documentation debt | Docs missing / wrong / outdated |
                        """
                    )

                gr.Textbox(
                    label="System status",
                    value=_startup_status(),
                    interactive=False,
                    lines=1,
                    max_lines=2,
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        title = gr.Textbox(
                            label="Issue title",
                            lines=2,
                            placeholder="e.g. FIXME: temporary hack in payment parser",
                        )
                        description = gr.Textbox(
                            label="Issue description",
                            lines=6,
                            placeholder="Describe the problem in plain language…",
                        )
                        mode = gr.Radio(
                            choices=[
                                "Title only",
                                "Description only",
                                "Title + Description",
                            ],
                            value="Title + Description",
                            label="Text for Classify",
                            info="Compare-all runs all three modes.",
                        )
                        with gr.Row():
                            run_btn = gr.Button(
                                "Classify",
                                variant="primary",
                                size="lg",
                            )
                            compare_btn = gr.Button(
                                "Compare Title / Desc / Both",
                                variant="secondary",
                                size="lg",
                            )
                        gr.Markdown("#### Keyword cues")
                        cues_view = gr.Markdown(
                            "_Pick a sample on the right — cue words highlight here._"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Result")
                        identification = gr.Textbox(
                            label="Step 1 — Identification",
                            interactive=False,
                            elem_classes=["result-card"],
                        )
                        category = gr.Textbox(
                            label="Step 2 — Category (short code)",
                            interactive=False,
                            elem_classes=["result-card"],
                        )
                        final_label = gr.Textbox(
                            label="Final label (short + full name)",
                            interactive=False,
                            elem_classes=["result-card"],
                        )
                        timing = gr.Textbox(label="Time taken", interactive=False)
                        detail = gr.Markdown()

                gr.Markdown("#### Confidence")
                with gr.Row():
                    id_plot = gr.BarPlot(
                        value=EMPTY_ID_PROBS,
                        x="label",
                        y="probability",
                        title="Identification — P(Not-SATD) vs P(SATD)",
                        y_lim=[0, 1],
                        height=200,
                    )
                    cat_plot = gr.BarPlot(
                        value=EMPTY_CAT_PROBS,
                        x="label",
                        y="probability",
                        title="Categorization — category probabilities",
                        y_lim=[0, 1],
                        height=200,
                    )

                gr.Markdown("### Title vs Description vs Both")
                compare_table = gr.Dataframe(
                    value=EMPTY_COMPARE,
                    label="Same issue · three input variants",
                    interactive=False,
                    wrap=True,
                )
                compare_note = gr.Markdown(
                    "_Click **Compare Title / Desc / Both** to fill this table._"
                )

                gr.Markdown(
                    f"""
                    ---
                    **Stack:** Qwen3-Embedding-0.6B → XGBoost identification → XGBoost categorization
                    (issue models, Pipeline A) · Models: `{default_model_dir()}`
                    """
                )

            # -------- Sample sidebar (right, sticky) --------
            with gr.Column(scale=3, min_width=280, elem_classes=["sample-sidebar"]):
                gr.Markdown("### Demo samples")
                gr.Markdown("Click one — it fills the input. Then Classify.")
                sample_buttons = []
                for label in SAMPLE_BUTTONS:
                    sample_buttons.append(
                        gr.Button(label, variant="secondary", size="sm")
                    )
                with gr.Accordion("Presenter cheat-sheet", open=False):
                    gr.Markdown(EXAMPLE_HINTS)

        run_btn.click(
            fn=classify,
            inputs=[title, description, mode],
            outputs=[
                detail,
                identification,
                category,
                final_label,
                id_plot,
                cat_plot,
                cues_view,
                timing,
            ],
        )
        compare_btn.click(
            fn=compare_modes,
            inputs=[title, description],
            outputs=[compare_table, compare_note, timing],
        )
        for comp in (title, description, mode):
            comp.change(
                fn=refresh_cues,
                inputs=[title, description, mode],
                outputs=[cues_view],
            )

        for i, btn in enumerate(sample_buttons):
            btn.click(
                fn=lambda idx=i: load_sample(idx),
                inputs=None,
                outputs=[title, description, mode, cues_view],
            )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
