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
/* Login and registration: a compact course card on the same digital desk. */
[data-testid="stAppViewContainer"]:has(.desk-auth-brand) {
    background-color: #e7dfcd;
    background-image:
        radial-gradient(circle at 18% 14%, rgba(255,253,244,.82), transparent 31%),
        radial-gradient(circle at 82% 88%, rgba(168,65,50,.055), transparent 26%),
        repeating-linear-gradient(
            0deg,
            transparent 0 31px,
            rgba(23,41,71,.026) 32px
        );
    background-size: auto;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-testid="stMainBlockContainer"] {
    background-color: var(--desk-paper);
    background-image:
        radial-gradient(circle at 12% 16%, rgba(71,57,35,.035) 0 .7px, transparent .9px),
        repeating-linear-gradient(0deg, transparent 0 31px, rgba(23,41,71,.022) 32px);
    border: 1px solid rgba(91,69,39,.16);
    border-radius: .7rem;
    box-shadow: 0 28px 80px rgba(50,39,25,.18);
    margin: clamp(3.2rem, 9vh, 6.5rem) auto 3rem;
    max-width: 560px;
    min-height: 0;
    overflow: hidden;
    padding: 3rem 3.35rem 3.2rem 4rem;
    position: relative;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-testid="stMainBlockContainer"]::before {
    background: rgba(168,65,50,.22);
    bottom: 0;
    content: "";
    left: 2.35rem;
    position: absolute;
    top: 0;
    width: 1px;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-testid="stMainBlockContainer"]::after {
    border: 1px solid rgba(168,65,50,.22);
    border-radius: 50%;
    content: "";
    height: 7rem;
    position: absolute;
    right: -4.2rem;
    top: -4.2rem;
    width: 7rem;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-testid="stExpandSidebarButton"] {
    display: none !important;
}
.st-key-desk_auth_shell {
    margin-bottom: 1.55rem;
}
.desk-auth-brand {
    position: relative;
}
.desk-auth-kicker {
    color: var(--desk-vermilion);
    font-size: .67rem;
    font-weight: 800;
    letter-spacing: .18em;
    margin-bottom: 1.15rem;
}
.desk-auth-heading {
    align-items: center;
    display: flex;
    gap: .95rem;
}
.desk-auth-seal {
    align-items: center;
    background: var(--desk-vermilion);
    border-radius: .22rem;
    box-shadow:
        inset 0 0 0 2px rgba(255,255,255,.3),
        0 6px 16px rgba(168,65,50,.2);
    color: #fffaf0;
    display: inline-flex;
    flex: 0 0 auto;
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: .76rem;
    height: 2.75rem;
    justify-content: center;
    letter-spacing: .08em;
    line-height: 1.05;
    text-align: center;
    transform: rotate(-2deg);
    width: 2.75rem;
}
.desk-auth-heading h1 {
    color: var(--desk-ink);
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: 1.72rem;
    font-weight: 760;
    letter-spacing: .06em;
    line-height: 1.15;
    margin: 0;
}
.desk-auth-heading p {
    color: var(--desk-ink-soft);
    font-size: .76rem;
    line-height: 1.6;
    margin: .34rem 0 0;
}
.desk-auth-note {
    border-left: 2px solid rgba(61,118,100,.42);
    color: var(--desk-jade);
    font-size: .7rem;
    letter-spacing: .035em;
    margin: 1.45rem 0 0;
    padding-left: .65rem;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-baseweb="tab-list"] {
    background: rgba(246,240,223,.72);
    border: 1px solid rgba(91,69,39,.11);
    border-radius: .48rem;
    display: flex;
    gap: .25rem;
    padding: .25rem;
    width: 100%;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-baseweb="tab"] {
    border-radius: .34rem;
    color: var(--desk-ink-soft);
    font-size: .78rem;
    flex: 1;
    font-weight: 680;
    height: 2.4rem;
    justify-content: center;
    padding: 0 1.25rem;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-baseweb="tab"][aria-selected="true"] {
    background: var(--desk-paper);
    box-shadow: 0 2px 8px rgba(50,39,25,.08);
    color: var(--desk-vermilion);
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-baseweb="tab-highlight"] {
    display: none;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-baseweb="tab-border"] {
    display: none;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-testid="stForm"] {
    border: 0;
    padding: 1.35rem 0 0;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-testid="stTextInput"] label {
    color: var(--desk-ink);
    font-size: .75rem;
    font-weight: 680;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-testid="stTextInput"] [data-baseweb="input"] {
    background: rgba(255,255,255,.7);
    border: 1px solid rgba(91,69,39,.16);
    border-radius: .42rem;
    box-shadow: inset 0 1px 2px rgba(50,39,25,.035);
    min-height: 2.9rem;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
    border-color: rgba(168,65,50,.55);
    box-shadow: 0 0 0 3px rgba(168,65,50,.08);
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-testid="stTextInput"] input {
    background: transparent;
    color: var(--desk-ink);
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-testid="stFormSubmitButton"] button {
    background: var(--desk-ink);
    border: 0;
    border-radius: .42rem;
    box-shadow: 0 8px 18px rgba(23,41,71,.16);
    color: #fffdf4;
    font-weight: 720;
    min-height: 2.9rem;
    transition: transform .16s ease, background .16s ease, box-shadow .16s ease;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-brand)
    [data-testid="stFormSubmitButton"] button:hover {
    background: var(--desk-vermilion);
    box-shadow: 0 10px 22px rgba(168,65,50,.2);
    color: #fffdf4;
    transform: translateY(-1px);
}
@media (max-width: 760px) {
    [data-testid="stAppViewContainer"]:has(.desk-auth-brand)
        [data-testid="stMainBlockContainer"] {
        margin: 1rem .75rem 2rem;
        max-width: none;
        padding: 2.2rem 1.35rem 2.4rem 2rem;
    }
    [data-testid="stAppViewContainer"]:has(.desk-auth-brand)
        [data-testid="stMainBlockContainer"]::before {
        left: 1.15rem;
    }
    .desk-auth-heading h1 {
        font-size: 1.42rem;
    }
    .desk-auth-heading p {
        font-size: .7rem;
    }
}
@media (max-width: 390px) {
    .desk-auth-heading {
        align-items: flex-start;
    }
    .desk-auth-note {
        line-height: 1.55;
    }
}

/* Auth v2: visual design follows the approved asymmetric editorial mockup. */
[data-testid="stAppViewContainer"]:has(.desk-auth-page-marker) {
    background-color: #f0eadf;
    background-image:
        radial-gradient(circle at 14% 18%, rgba(255,255,255,.92), transparent 28%),
        radial-gradient(circle at 86% 78%, rgba(183,71,54,.055), transparent 24%),
        linear-gradient(112deg, rgba(255,253,246,.72), rgba(232,222,203,.48));
    background-size: auto;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-page-marker)
    [data-testid="stMainBlockContainer"] {
    margin: 0 auto;
    max-width: 1440px;
    min-height: 100vh;
    padding: clamp(2.5rem, 6vh, 5.2rem) clamp(2rem, 6vw, 6.5rem);
}
[data-testid="stAppViewContainer"]:has(.desk-auth-page-marker)
    [data-testid="stMainBlockContainer"] > div {
    min-height: calc(100vh - clamp(5rem, 12vh, 10.4rem));
}
[data-testid="stAppViewContainer"]:has(.desk-auth-page-marker)
    [data-testid="stHorizontalBlock"] {
    align-items: center;
    display: grid;
    gap: clamp(3rem, 7vw, 8rem);
    grid-template-columns: minmax(0, 1.12fr) minmax(420px, .88fr);
    min-height: inherit;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-page-marker)
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: initial !important;
    min-width: 0;
    width: auto !important;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-page-marker)
    [data-testid="stColumn"]:has(.desk-auth-panel-header) {
    background: rgba(255,253,246,.94);
    border: 1px solid rgba(96,73,43,.14);
    border-radius: .85rem;
    box-shadow:
        0 28px 70px rgba(53,41,24,.16),
        0 3px 9px rgba(53,41,24,.06);
    max-width: 470px;
    overflow: hidden;
    padding: 3.15rem 3.1rem 2.6rem;
    position: relative;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-page-marker)
    [data-testid="stColumn"]:has(.desk-auth-panel-header)::before {
    background: linear-gradient(
        90deg,
        transparent,
        rgba(183,71,54,.92) 42% 58%,
        transparent
    );
    content: "";
    height: 2px;
    left: 32%;
    position: absolute;
    right: 32%;
    top: 5.2rem;
}
[data-testid="stAppViewContainer"]:has(.desk-auth-page-marker)
    [data-testid="stExpandSidebarButton"] {
    display: none !important;
}
.desk-auth-hero {
    max-width: 660px;
    padding-top: .5rem;
}
.desk-auth-wordmark {
    align-items: center;
    color: var(--desk-ink);
    display: flex;
    font-family: Georgia, "Times New Roman", serif;
    font-size: .86rem;
    font-weight: 720;
    gap: .75rem;
    letter-spacing: .12em;
}
.desk-auth-mini-seal {
    align-items: center;
    border: 2px solid var(--desk-vermilion);
    border-radius: .18rem;
    color: var(--desk-vermilion);
    display: inline-flex;
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: .62rem;
    font-weight: 800;
    height: 2.15rem;
    justify-content: center;
    letter-spacing: -.04em;
    transform: rotate(-2deg);
    width: 2.15rem;
}
.desk-auth-hero h1 {
    color: var(--desk-ink);
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: clamp(3rem, 5.1vw, 5rem);
    font-weight: 760;
    letter-spacing: .025em;
    line-height: 1.08;
    margin: 1.6rem 0 .85rem;
}
.desk-auth-title-rule {
    background: var(--desk-vermilion);
    border-radius: 999px;
    height: 3px;
    margin-bottom: 1.15rem;
    width: 3.2rem;
}
.desk-auth-tagline {
    color: var(--desk-ink-soft);
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: 1rem;
    letter-spacing: .045em;
    line-height: 1.8;
    margin: 0;
}
.desk-auth-collage {
    height: 320px;
    margin-top: 2.1rem;
    position: relative;
}
.desk-auth-paper {
    background-color: #fffdf6;
    background-image:
        repeating-linear-gradient(0deg, transparent 0 25px, rgba(49,95,196,.07) 26px);
    border: 1px solid rgba(96,73,43,.12);
    box-shadow: 0 18px 34px rgba(53,41,24,.12);
    position: absolute;
}
.desk-auth-paper-back {
    height: 220px;
    left: 8.5rem;
    top: 2.3rem;
    transform: rotate(7deg);
    width: min(68%, 410px);
}
.desk-auth-paper-back span {
    color: rgba(23,42,74,.35);
    font-size: .67rem;
    left: 1.3rem;
    letter-spacing: .16em;
    position: absolute;
    top: 1.2rem;
}
.desk-auth-paper-main {
    height: 230px;
    left: 4.5rem;
    overflow: hidden;
    padding: 1.7rem 2rem;
    top: 1.2rem;
    transform: rotate(-3deg);
    width: min(70%, 430px);
}
.desk-auth-paper-main small {
    color: var(--desk-vermilion);
    display: block;
    font-size: .58rem;
    letter-spacing: .14em;
}
.desk-auth-paper-main strong {
    color: var(--desk-ink);
    display: block;
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: 1.16rem;
    margin-top: .85rem;
}
.desk-auth-paper-main p {
    color: rgba(23,42,74,.55);
    font-size: .63rem;
    letter-spacing: .04em;
    margin-top: .45rem;
}
.desk-auth-handnote {
    color: #3567cf;
    font-family: "KaiTi", "STKaiti", cursive;
    font-size: .82rem;
    left: -1.4rem;
    line-height: 1.75;
    position: absolute;
    top: 5.8rem;
    transform: rotate(-7deg);
    z-index: 4;
}
.desk-auth-bookmark {
    background: #456d5e;
    box-shadow: 0 7px 16px rgba(30,68,55,.2);
    color: #fff9e9;
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: .62rem;
    height: 8.4rem;
    letter-spacing: .1em;
    padding-top: 2rem;
    position: absolute;
    right: 4rem;
    text-align: center;
    top: 7rem;
    transform: rotate(11deg);
    width: 2.15rem;
    writing-mode: vertical-rl;
    z-index: 5;
}
.desk-auth-bookmark::after {
    border-left: 1.075rem solid transparent;
    border-right: 1.075rem solid transparent;
    border-top: 1rem solid #456d5e;
    bottom: -1rem;
    content: "";
    left: 0;
    position: absolute;
}
.desk-auth-geometry {
    border: 1px solid rgba(49,95,196,.28);
    border-radius: 50%;
    bottom: 1.3rem;
    height: 5.2rem;
    position: absolute;
    right: 2.1rem;
    width: 5.2rem;
}
.desk-auth-geometry::before,
.desk-auth-geometry::after {
    background: rgba(49,95,196,.24);
    content: "";
    height: 1px;
    left: -.6rem;
    position: absolute;
    top: 50%;
    width: 6.4rem;
}
.desk-auth-geometry::after {
    transform: rotate(60deg);
}
.desk-auth-geometry i {
    background: #3567cf;
    border-radius: 50%;
    height: .26rem;
    position: absolute;
    width: .26rem;
}
.desk-auth-geometry i:nth-child(1) { left: 48%; top: -.13rem; }
.desk-auth-geometry i:nth-child(2) { bottom: .45rem; left: .6rem; }
.desk-auth-geometry i:nth-child(3) { bottom: .45rem; right: .6rem; }
.desk-auth-panel-header {
    text-align: center;
}
.desk-auth-panel-kicker {
    color: var(--desk-ink);
    font-family: Georgia, "Times New Roman", serif;
    font-size: .66rem;
    font-weight: 720;
    letter-spacing: .14em;
    margin-bottom: 2.8rem;
}
.desk-auth-panel-header h2 {
    color: var(--desk-ink);
    font-family: "Noto Serif SC", "Songti SC", STSong, serif;
    font-size: 1.75rem;
    letter-spacing: .08em;
    margin: 0;
}
.desk-auth-panel-header p {
    color: var(--desk-ink-soft);
    font-size: .72rem;
    margin: .6rem 0 1.7rem;
}
[data-testid="stColumn"]:has(.desk-auth-panel-header)
    [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(96,73,43,.16);
    display: flex;
    gap: 0;
    padding: 0;
    width: 100%;
}
[data-testid="stColumn"]:has(.desk-auth-panel-header)
    [data-baseweb="tab"] {
    color: var(--desk-ink-soft);
    flex: 1;
    font-size: .78rem;
    font-weight: 680;
    height: 2.6rem;
    justify-content: center;
}
[data-testid="stColumn"]:has(.desk-auth-panel-header)
    [data-baseweb="tab"][aria-selected="true"] {
    color: var(--desk-vermilion);
}
[data-testid="stColumn"]:has(.desk-auth-panel-header)
    [data-baseweb="tab-highlight"] {
    background: var(--desk-vermilion);
    height: 2px;
}
[data-testid="stColumn"]:has(.desk-auth-panel-header)
    [data-testid="stForm"] {
    border: 0;
    padding: 1.35rem 0 .2rem;
}
[data-testid="stColumn"]:has(.desk-auth-panel-header)
    [data-testid="stTextInput"] label {
    color: var(--desk-ink);
    font-size: .72rem;
    font-weight: 680;
}
[data-testid="stColumn"]:has(.desk-auth-panel-header)
    [data-testid="stTextInput"] [data-baseweb="input"] {
    background: #fffef9;
    border: 1px solid rgba(70,71,74,.2);
    border-radius: .35rem;
    box-shadow: none;
    min-height: 3rem;
}
[data-testid="stColumn"]:has(.desk-auth-panel-header)
    [data-testid="stTextInput"] input {
    background: transparent;
    color: var(--desk-ink);
}
[data-testid="stColumn"]:has(.desk-auth-panel-header)
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
    border-color: rgba(49,95,196,.6);
    box-shadow: 0 0 0 3px rgba(49,95,196,.09);
}
[data-testid="stColumn"]:has(.desk-auth-panel-header)
    [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(100deg, #142a4a, #18365e);
    border: 0;
    border-radius: .35rem;
    box-shadow: 0 9px 20px rgba(20,42,74,.18);
    color: #fffdf6;
    font-weight: 720;
    letter-spacing: .08em;
    min-height: 3.05rem;
    position: relative;
}
[data-testid="stColumn"]:has(.desk-auth-panel-header)
    [data-testid="stFormSubmitButton"] button:hover {
    background: #315fc4;
    box-shadow: 0 11px 25px rgba(49,95,196,.23);
    color: #fff;
    transform: translateY(-1px);
}
.desk-auth-privacy {
    color: rgba(23,42,74,.48);
    font-size: .62rem;
    margin-top: .8rem;
    text-align: center;
}
@media (max-width: 980px) {
    [data-testid="stAppViewContainer"]:has(.desk-auth-page-marker)
        [data-testid="stHorizontalBlock"] {
        gap: 2.2rem;
        grid-template-columns: minmax(0, 1fr) minmax(380px, .9fr);
    }
    .desk-auth-hero h1 {
        font-size: 3rem;
    }
}
@media (max-width: 760px) {
    [data-testid="stAppViewContainer"]:has(.desk-auth-page-marker)
        [data-testid="stMainBlockContainer"] {
        padding: 1.5rem 1rem 3rem;
    }
    [data-testid="stAppViewContainer"]:has(.desk-auth-page-marker)
        [data-testid="stHorizontalBlock"] {
        display: block;
        min-height: 0;
    }
    .desk-auth-hero {
        margin: 0 auto 1.8rem;
        max-width: 470px;
        padding: .4rem .35rem 0;
    }
    .desk-auth-hero h1 {
        font-size: 2.45rem;
        margin-top: 1.1rem;
    }
    .desk-auth-collage {
        display: none;
    }
    [data-testid="stAppViewContainer"]:has(.desk-auth-page-marker)
        [data-testid="stColumn"]:has(.desk-auth-panel-header) {
        margin: 0 auto;
        max-width: 470px;
        padding: 2.5rem 1.5rem 2.2rem;
    }
}
/* Keep Streamlit's sidebar control usable without leaving a white hosting bar. */
[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.35rem;
    pointer-events: none;
}
[data-testid="stHeader"] button,
[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"],
[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {
    pointer-events: auto;
}

/* Streamlit's own application controls stay visible; only its hosting chrome is removed. */
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
#MainMenu,
footer {
    display: none !important;
}
[data-testid="stToolbar"] {
    background: transparent !important;
    pointer-events: none;
}
[data-testid="stExpandSidebarButton"] {
    align-items: center;
    background: rgba(255,253,244,.96) !important;
    border: 1px solid rgba(94,74,46,.18) !important;
    border-radius: .55rem !important;
    box-shadow: 0 7px 22px rgba(50,39,25,.14) !important;
    color: var(--desk-vermilion) !important;
    display: flex !important;
    height: 2.45rem;
    justify-content: center;
    left: .85rem;
    pointer-events: auto !important;
    position: fixed;
    top: .72rem;
    width: 2.45rem;
    z-index: 1000000;
}
[data-testid="stExpandSidebarButton"]:hover {
    background: #fff9e9 !important;
    border-color: rgba(168,65,50,.34) !important;
    color: var(--desk-ink) !important;
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
    max-width: 1240px;
    padding: 1.15rem 2.2rem 9.5rem;
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
.desk-brand-header {
    display: none;
}
[data-testid="stElementContainer"]:has(.desk-brand-header),
[data-testid="stElementContainer"]:has(.desk-brand-header) + [data-testid="stElementContainer"] {
    display: none;
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

@property --desk-thinking-seconds {
    syntax: "<integer>";
    initial-value: 0;
    inherits: false;
}
@keyframes desk-thinking-clock {
    from { --desk-thinking-seconds: 0; }
    to { --desk-thinking-seconds: 900; }
}
.desk-live-thinking-marker {
    display: none;
}
[data-testid="stExpander"]:has(.desk-live-thinking-marker) summary p::after,
[data-testid="stExpander"]:has(.desk-live-thinking-marker) button p::after {
    animation: desk-thinking-clock 900s steps(900, end) forwards;
    color: var(--desk-ink-soft);
    content: "（已用时 " counter(deskThinkingSeconds) " 秒）";
    counter-reset: deskThinkingSeconds var(--desk-thinking-seconds);
    display: inline-block;
    font-variant-numeric: tabular-nums;
    margin-left: .35rem;
    min-width: 7.4em;
}

[data-testid="stBottomBlockContainer"] {
    background: linear-gradient(180deg, transparent, rgba(232,225,209,.96) 38%);
    padding: .5rem 2rem .72rem;
}
[data-testid="stBottom"] {
    background: transparent !important;
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
    padding: .58rem .78rem .5rem;
}
.st-key-teacher_composer_actions > div[data-testid="stVerticalBlock"] {
    gap: .46rem;
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
    margin-right: 0;
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
    align-items: center;
    color: var(--desk-ink-soft);
    display: flex;
    font-size: .65rem;
    gap: .42rem;
    justify-content: flex-end;
    letter-spacing: .035em;
    pointer-events: none;
}
.desk-mode-chip {
    background: rgba(61,118,100,.08);
    border: 1px solid rgba(61,118,100,.2);
    border-radius: 999px;
    color: var(--desk-jade);
    padding: .24rem .55rem;
    white-space: nowrap;
}
.desk-mode-chip.is-auto {
    background: rgba(168,65,50,.055);
    border-color: rgba(168,65,50,.16);
    color: var(--desk-vermilion);
}
.st-key-teacher_composer_controls [data-testid="stHorizontalBlock"] {
    align-items: center;
    gap: .6rem;
}
.st-key-teacher_composer_controls [data-testid="stColumn"] {
    min-width: 0;
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
        padding: .35rem .55rem .48rem;
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
    .desk-mode-chip {
        padding: .2rem .42rem;
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


def render_thinking_sections(
    turn: object,
    expanded: bool = True,
    *,
    container: object | None = None,
) -> None:
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
    target = container or st
    target.markdown(
        '<aside class="desk-thinking-rail"><div class="desk-thinking-rail-title">备课批注</div>'
        + "".join(cards)
        + "</aside>",
        unsafe_allow_html=True,
    )
