"""Visual system for the digital teacher desk Streamlit interface."""

from __future__ import annotations

import streamlit as st
from html import escape


DESK_THEME_CSS = """
<style>
:root {
    --desk-paper: #fffdf4;
    --desk-paper-deep: #f6f0df;
    --desk-ivory: #f2ead7;
    --desk-ink: #172947;
    --desk-ink-soft: #5f6b78;
    --desk-vermilion: #a84132;
    --desk-jade: #3d7664;
    --desk-rule: rgba(168, 65, 50, 0.2);
    --desk-shadow: 0 22px 60px rgba(50, 39, 25, 0.12);
}

html,
body,
[data-testid="stAppViewContainer"] {
    background-color: #e8e1d1;
    color: var(--desk-ink);
}
[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(circle at 16% 12%, rgba(255,255,255,.7) 0 1px, transparent 1.5px),
        linear-gradient(115deg, rgba(121,96,58,.035) 25%, transparent 25% 75%, rgba(121,96,58,.025) 75%);
    background-size: 19px 19px, 31px 31px;
}

/* Streamlit's own application controls stay visible; only its hosting chrome is removed. */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
    #MainMenu,
footer {
    display: none !important;
}

[data-testid="stSidebar"] {
    background: var(--desk-ivory);
    border-right: 1px solid rgba(78, 63, 42, 0.12);
    min-width: 288px;
    width: 288px;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background-image:
        linear-gradient(90deg, transparent 31px, rgba(168,65,50,.13) 32px, transparent 33px),
        repeating-linear-gradient(0deg, transparent 0 27px, rgba(23,41,71,.035) 28px);
    padding: 1.35rem 1rem 1.2rem;
}
.desk-directory {
    color: var(--desk-ink);
    margin: 0 0 .8rem;
}
.desk-directory-kicker {
    color: var(--desk-vermilion);
    font-size: .68rem;
    font-weight: 750;
    letter-spacing: .18em;
    text-transform: uppercase;
}
.desk-directory-title {
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: .08em;
    margin-top: .15rem;
}
.desk-history-time {
    color: rgba(23,41,71,.5);
    font-size: .66rem;
    margin: -.48rem .7rem .38rem;
    text-align: right;
}
[data-testid="stSidebar"] h3 {
    color: var(--desk-vermilion);
    font-size: .7rem;
    font-weight: 720;
    letter-spacing: .14em;
    margin: 1.1rem 0 .28rem;
}
[data-testid="stSidebar"] .stButton > button {
    border: 0;
    border-radius: .3rem;
    box-shadow: none;
    color: var(--desk-ink);
    font-size: .82rem;
    justify-content: flex-start;
    line-height: 1.35;
    min-height: 2.35rem;
    padding: .48rem .72rem;
    text-align: left;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,253,244,.72);
    color: var(--desk-vermilion);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--desk-paper);
    box-shadow: inset 3px 0 var(--desk-vermilion), 0 3px 12px rgba(67,49,26,.07);
    color: var(--desk-ink);
    font-weight: 680;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1160px;
    padding: 1.45rem 2.2rem 11rem;
}
.desk-brand-header {
    align-items: center;
    display: flex;
    gap: .75rem;
    margin: 0 auto 1rem;
    max-width: 1020px;
}
.desk-brand-seal {
    align-items: center;
    background: var(--desk-vermilion);
    border-radius: .22rem;
    box-shadow: inset 0 0 0 2px rgba(255,255,255,.34);
    color: #fffaf0;
    display: inline-flex;
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: .74rem;
    height: 2.3rem;
    justify-content: center;
    letter-spacing: .08em;
    line-height: 1.05;
    text-align: center;
    width: 2.3rem;
}
.desk-brand-name {
    color: var(--desk-ink);
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: 1.18rem;
    font-weight: 750;
    letter-spacing: .08em;
}
.desk-brand-tagline {
    color: var(--desk-ink-soft);
    font-size: .7rem;
    letter-spacing: .06em;
    margin-top: .08rem;
}

.desk-paper-sheet,
[class*="st-key-desk_paper_sheet_"] {
    background-color: var(--desk-paper);
    background-image:
        radial-gradient(circle at 14% 18%, rgba(71,57,35,.035) 0 .7px, transparent .9px),
        repeating-linear-gradient(0deg, transparent 0 31px, rgba(23,41,71,.025) 32px);
    border: 1px solid rgba(94, 74, 46, .13);
    border-radius: .45rem;
    box-shadow: var(--desk-shadow);
    margin: 0 auto;
    max-width: 1020px;
    min-height: 48vh;
    overflow: hidden;
    padding: 2.2rem 3rem 3.4rem 4.5rem;
    position: relative;
}
[class*="st-key-desk_paper_sheet_"]::before {
    background: var(--desk-rule);
    bottom: 0;
    content: "";
    left: 3rem;
    position: absolute;
    top: 0;
    width: 1px;
}
.desk-lesson-heading {
    border-bottom: 1px solid rgba(23,41,71,.12);
    margin: 0 0 1.7rem;
    padding: 0 0 .85rem;
}
.desk-lesson-title {
    color: var(--desk-ink);
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: 1.18rem;
    font-weight: 720;
    letter-spacing: .035em;
}
.desk-lesson-meta {
    color: var(--desk-ink-soft);
    font-size: .7rem;
    letter-spacing: .06em;
    margin-top: .22rem;
}

[data-testid="stChatMessage"] {
    background: transparent;
    border: 0;
    padding: .55rem 0 1.15rem;
}
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {
    display: none;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    padding: 0;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #fff2b9;
    border: 1px solid rgba(122,91,35,.12);
    border-radius: .2rem .2rem .7rem .2rem;
    box-shadow: 0 7px 18px rgba(74,57,29,.1);
    margin: .2rem 0 1.8rem auto;
    max-width: 72%;
    padding: .82rem 1rem .75rem;
    transform: rotate(.25deg);
    width: fit-content;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])::before {
    background: rgba(168,65,50,.74);
    border-radius: 50%;
    box-shadow: 0 1px 2px rgba(0,0,0,.2);
    content: "";
    height: .48rem;
    position: absolute;
    right: .7rem;
    top: .35rem;
    width: .48rem;
}
.desk-user-note {
    background: #fff2b9;
    border: 1px solid rgba(122,91,35,.12);
    border-radius: .2rem .2rem .7rem .2rem;
    box-shadow: 0 7px 18px rgba(74,57,29,.1);
    color: var(--desk-ink);
    margin: .2rem 0 1.8rem auto;
    max-width: 72%;
    padding: .82rem 1rem .75rem;
    position: relative;
    transform: rotate(.25deg);
    width: fit-content;
}
.desk-user-note::before {
    background: rgba(168,65,50,.74);
    border-radius: 50%;
    box-shadow: 0 1px 2px rgba(0,0,0,.2);
    content: "";
    height: .48rem;
    position: absolute;
    right: .7rem;
    top: .35rem;
    width: .48rem;
}
.desk-answer,
[class*="st-key-desk_answer_"] {
    color: var(--desk-ink);
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: 1rem;
    line-height: 1.95;
    min-width: 0;
}
[class*="st-key-desk_answer_"] p {
    line-height: 1.95;
    margin-bottom: 1rem;
}
[class*="st-key-desk_answer_"] h1,
[class*="st-key-desk_answer_"] h2,
[class*="st-key-desk_answer_"] h3 {
    color: var(--desk-ink);
    font-family: inherit;
    letter-spacing: .025em;
}

.desk-thinking-rail {
    border-left: 2px solid rgba(168,65,50,.35);
    color: var(--desk-ink-soft);
    font-size: .72rem;
    margin: .25rem 0 .4rem;
    padding: .1rem 0 .1rem .72rem;
}
.desk-thinking-card { border-bottom: 1px dashed rgba(23,41,71,.13); margin: 0 0 .72rem; padding: 0 0 .65rem; }
.desk-thinking-card:last-child { border-bottom: 0; margin-bottom: 0; }
.desk-thinking-card-title { color: var(--desk-ink); font-weight: 740; letter-spacing: .09em; }
.desk-thinking-card-summary { color: var(--desk-ink-soft); line-height: 1.55; margin: .3rem 0; }
.desk-thinking-card-detail { line-height: 1.5; margin: .16rem 0; }
.st-key-desk_compat_question_1,
[class*="st-key-desk_compat_question_"] {
    display: none;
}
.desk-thinking-rail.is-collapsed {
    border-left-color: rgba(61,118,100,.45);
}
.desk-thinking-eyebrow {
    color: var(--desk-vermilion);
    font-size: .66rem;
    font-weight: 750;
    letter-spacing: .15em;
    margin-bottom: .65rem;
}
.desk-thinking-section {
    border-bottom: 1px dashed rgba(23,41,71,.13);
    margin: 0 0 .72rem;
    padding: 0 0 .65rem;
}
.desk-thinking-section:last-child {
    border-bottom: 0;
    margin-bottom: 0;
}
.desk-thinking-title {
    color: var(--desk-ink);
    font-weight: 740;
    letter-spacing: .09em;
}
.desk-thinking-stamp {
    border: 1px solid currentColor;
    border-radius: 999px;
    color: var(--desk-jade);
    float: right;
    font-size: .58rem;
    line-height: 1.35;
    padding: 0 .3rem;
}
.desk-thinking-stamp.is-warning {
    color: var(--desk-vermilion);
}
.desk-thinking-summary {
    color: var(--desk-ink-soft);
    line-height: 1.55;
    margin: .3rem 0;
}
.desk-thinking-details {
    list-style: none;
    margin: .28rem 0 0;
    padding: 0;
}
.desk-thinking-details li {
    line-height: 1.5;
    margin: .16rem 0;
    padding-left: .68rem;
    position: relative;
}
.desk-thinking-details li::before {
    color: var(--desk-vermilion);
    content: "·";
    left: 0;
    position: absolute;
}
[data-testid="stExpander"]:has(.desk-thinking-rail) {
    background: rgba(246,240,223,.58);
    border: 0;
    border-radius: .3rem;
    box-shadow: none;
}
[data-testid="stExpander"]:has(.desk-thinking-rail) summary {
    color: var(--desk-ink);
    font-size: .73rem;
}

[data-testid="stBottomBlockContainer"] {
    background: linear-gradient(180deg, transparent, rgba(232,225,209,.96) 25%);
    padding: 1.4rem 2rem .8rem;
}
.desk-composer,
.st-key-teacher_composer_actions {
    background-color: var(--desk-paper);
    background-image: linear-gradient(90deg, rgba(168,65,50,.08) 1px, transparent 1px);
    background-size: 28px 100%;
    border: 1px solid rgba(91,69,39,.14);
    border-radius: .65rem;
    box-shadow: 0 15px 38px rgba(49,38,23,.16);
    margin: 0 auto;
    max-width: 920px;
    padding: .7rem .85rem .55rem;
}
.st-key-teacher_composer_actions > div[data-testid="stVerticalBlock"] {
    gap: .42rem;
}
.st-key-teacher_composer_actions [data-testid="stChatInput"] {
    background: rgba(255,255,255,.54);
    border: 0;
    border-bottom: 1px solid rgba(23,41,71,.14);
    border-radius: .3rem;
    box-shadow: none;
}
.st-key-teacher_composer_actions [data-testid="stChatInputSubmitButton"] {
    background: var(--desk-vermilion);
    border-radius: 50%;
    color: #fffdf4;
    height: 2.55rem;
    margin-right: .2rem;
    width: 2.55rem;
}
.st-key-context_usage_control {
    color: var(--desk-ink-soft);
    margin-right: 3.3rem;
    position: static;
    transform: none;
    width: max-content !important;
}
.st-key-context_usage_control [data-testid="stPopover"] > button {
    background: rgba(23,41,71,.045);
    border: 0;
    border-radius: 999px;
    color: var(--desk-ink-soft);
    font-size: .68rem;
    min-height: 1.8rem;
    padding: .2rem .55rem;
}
.desk-composer-status {
    color: var(--desk-ink-soft);
    font-size: .65rem;
    letter-spacing: .035em;
    pointer-events: none;
}
.desk-composer-dot {
    background: var(--desk-jade);
    border-radius: 50%;
    display: inline-block;
    height: .38rem;
    margin: 0 .26rem 0 .55rem;
    vertical-align: .03rem;
    width: .38rem;
}

@media (max-width: 760px) {
    [data-testid="stSidebar"] {
        min-width: min(82vw, 288px);
        width: min(82vw, 288px);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 100%;
        padding: 1rem .8rem 10.5rem;
    }
    .desk-brand-header {
        padding: 0 .2rem;
    }
    .desk-paper-sheet,
    [class*="st-key-desk_paper_sheet_"] {
        border-radius: .25rem;
        min-height: 40vh;
        padding: 1.35rem 1rem 2.4rem;
    }
    [class*="st-key-desk_paper_sheet_"]::before {
        display: none;
    }
    [data-testid="stHorizontalBlock"]:has(.desk-thinking-rail) {
        display: block;
    }
    [data-testid="stHorizontalBlock"]:has(.desk-thinking-rail) > [data-testid="stColumn"] {
        min-width: 100% !important;
        width: 100% !important;
    }
    .desk-thinking-rail {
        border-left: 0;
        border-top: 1px solid rgba(168,65,50,.3);
        margin-bottom: 1rem;
        padding: .65rem .2rem 0;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        max-width: 88%;
    }
    [data-testid="stBottomBlockContainer"] {
        padding-left: .55rem;
        padding-right: .55rem;
    }
    .desk-composer,
    .st-key-teacher_composer_actions {
        max-width: 100%;
    }
}

@media (max-width: 390px) {
    [data-testid="stMainBlockContainer"] {
        overflow-x: hidden;
        padding-left: .45rem;
        padding-right: .45rem;
    }
    .desk-paper-sheet,
    [class*="st-key-desk_paper_sheet_"] {
        padding-left: .75rem;
        padding-right: .75rem;
    }
    .desk-brand-tagline {
        display: none;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        max-width: 94%;
    }
    .desk-composer,
    .st-key-teacher_composer_actions {
        padding: .52rem .55rem .42rem;
    }
    .st-key-context_usage_control {
        margin-right: 2.8rem;
        max-width: calc(100% - 3rem);
    }
    .st-key-context_usage_control [data-testid="stPopover"] > button p {
        display: none;
    }
    .st-key-context_usage_control [data-testid="stPopover"] > button::after {
        content: "占用";
    }
    .desk-composer-status {
        font-size: .59rem;
        letter-spacing: 0;
        white-space: nowrap;
    }
}
</style>
"""


def render_brand_header() -> None:
    """Render the static product mark without exposing dynamic HTML."""

    st.markdown(
        """
        <header class="desk-brand-header">
          <span class="desk-brand-seal" aria-hidden="true">师<br>案</span>
          <span>
            <span class="desk-brand-name">数字教师书桌</span>
            <span class="desk-brand-tagline">把每次提问，整理成一页可继续学习的讲义</span>
          </span>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_thinking_sections(turn: object, expanded: bool = True) -> None:
    """Render the public four-slot teaching summary in the paper margin."""

    sections = getattr(turn, "sections", []) or []
    if not sections:
        return
    cards = []
    for section in sections:
        status = escape(str(getattr(section, "status", "completed")).split(".")[-1])
        details = "".join(
            f'<p class="desk-thinking-card-detail">· {escape(str(detail))}</p>'
            for detail in getattr(section, "details", [])
        )
        cards.append(
            f'<article class="desk-thinking-card" data-status="{status}">'
            f'<div class="desk-thinking-card-title">{escape(str(section.title))}</div>'
            f'<div class="desk-thinking-card-summary">{escape(str(section.summary))}</div>{details}</article>'
        )
    st.markdown(
        '<aside class="desk-thinking-rail"><div class="desk-thinking-rail-title">备课批注</div>'
        + "".join(cards)
        + "</aside>",
        unsafe_allow_html=True,
    )
