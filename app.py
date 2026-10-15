import os
import json
import re
import math
import ssl
from collections import Counter
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import textstat  
from spellchecker import SpellChecker
import easyocr 
import cv2
import numpy as np

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="AI-powered essay analyser with 8 dimensions: readability, vocabulary, structure, spelling, transitions, active voice, argument strength, and coherence.">
    <title>SunGrade AI — Deep Writing Analysis</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg:        #07101e
    --surface:   rgba(13, 22, 45, 0.80);
    --surface-2: rgba(18, 30, 60, 0.70);
    --border:    rgba(255,255,255,0.07);
    --border-2:  rgba(255,255,255,0.12);
    --accent:    #4f8ef7;
     --accent-2:  #8b5cf6;
    --accent-3:  #34d399;
    --green:     #22c55e;
    --yellow:    #eab308;
    --red:       #ef4444;
    --text-1:    #f1f5f9;
    --text-2:    #94a3b8;
    --text-3:    #64748b;
    --radius:    18px;
    --radius-sm: 12px;
    --shadow:    0 8px 32px rgba(0,0,0,0.55);
    --shadow-lg: 0 16px 60px rgba(0,0,0,0.65);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text-1);
    min-height: 100vh;
    overflow-x: hidden;
    padding: 2rem 1rem 5rem;
}

.bg-orbs { position: fixed; inset: 0; pointer-events: none; z-index: 0; }
.orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(110px);
    opacity: 0.15;
    animation: drift 20s ease-in-out infinite alternate;
}
.orb-1 { width: 600px; height: 600px; background: #4f8ef7; top: -200px; left: -200px; }
.orb-2 { width: 450px; height: 450px; background: #8b5cf6; bottom: -100px; right: -100px; animation-delay: -7s; }
.orb-3 { width: 350px; height: 350px; background: #22c55e; top: 55%; left: 55%; transform: translate(-50%,-50%); animation-delay: -14s; }

@keyframes drift {
    from { transform: translate(0, 0) scale(1); }
     to   { transform: translate(50px, 35px) scale(1.05); }
}

.app-wrapper {
    position: relative;
    z-index: 1;
    max-width: 960px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 1.75rem;
}

.app-header { text-align: center; padding: 0.5rem 0 0.25rem; }

.logo {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--surface);
    border: 1px solid var(--border-2);
    padding: 0.4rem 1.1rem;
    border-radius: 100px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 1.25rem;
    letter-spacing: 0.02em;
}
.logo-icon { font-size: 1rem; }
.logo-text  { color: var(--accent); }
.logo-version {
    background: rgba(79,142,247,0.15);
    color: var(--accent);
    font-size: 0.65rem;
    padding: 0.1rem 0.45rem;
    border-radius: 100px;
    border: 1px solid rgba(79,142,247,0.25);
    font-weight: 700;
    letter-spacing: 0.05em;
}

.app-title {
    font-size: clamp(2rem, 5.5vw, 3rem);
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
     -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.85rem;
    }

.app-subtitle {
    color: var(--text-2);
    font-size: 1rem;
     max-width: 520px;
    margin: 0 auto;
    line-height: 1.55;
}

.upload-section {
    background: var(--surface);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0;
     box-shadow: var(--shadow);
    overflow: hidden;
}

.tab-bar {
display: flex;
    border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.2);
}

.tab-btn {
    flex: 1;
    padding: 1rem 1.25rem;
    background: transparent;
    border: none;
    color: var(--text-3);
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
    letter-spacing: 0.01em;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}
.tab-btn:hover { color: var(--text-2); }
.tab-btn.active {
color: var(--accent);
    background: rgba(79,142,247,0.06);
}
.tab-btn.active::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    border-radius: 2px 2px 0 0;
}
.tab-icon { font-size: 1rem; }

.tab-panel { padding: 1.75rem; }
.tab-panel.hidden { display: none; }

.dropzone {
    border: 2px dashed rgba(79,142,247,0.35);
     border-radius: var(--radius-sm);
    padding: 3rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
    background: rgba(79,142,247,0.03);
}
.dropzone:hover, .dropzone.dragover {
    border-color: var(--accent);
    background: rgba(79,142,247,0.07);
    transform: scale(1.01);
}
.dz-icon { font-size: 2.8rem; margin-bottom: 0.9rem; }
.dz-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.45rem; }
.dz-subtitle { color: var(--text-2); font-size: 0.9rem; }

.preview-container {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}
.preview-container.hidden { display: none; }
#imagePreview {
    max-height: 180px;
    border-radius: 10px;
    border: 1px solid var(--border);
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    flex: 1;
    min-width: 0;
    object-fit: contain;
}

.text-input-wrapper { display: flex; flex-direction: column; gap: 0.6rem; }
.essay-textarea {
 width: 100%;
    min-height: 200px;
    background: rgba(0,0,0,0.25);
    border: 1.5px solid rgba(255,255,255,0.08);
    border-radius: var(--radius-sm);
    padding: 1.1rem 1.25rem;
    color: var(--text-1);
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    line-height: 1.65;
    resize: vertical;
    transition: border-color 0.2s ease;
    outline: none;
}
.essay-textarea:focus {
border-color: rgba(79,142,247,0.5);
    box-shadow: 0 0 0 3px rgba(79,142,247,0.08);
}
.essay-textarea::placeholder { color: var(--text-3); }
.textarea-footer {
display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 0.25rem;
}
.char-count { font-size: 0.78rem; color: var(--text-3); font-variant-numeric: tabular-nums; }

.action-bar {
    padding: 0 1.75rem 1.75rem;
    display: flex;
    gap: 1rem;
}

.btn-primary {
    flex: 1;
     padding: 1rem 1.5rem;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    border: none;
    border-radius: var(--radius-sm);
    color: white;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    box-shadow: 0 4px 22px rgba(79,142,247,0.35);
    letter-spacing: 0.02em;
}
.btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 32px rgba(79,142,247,0.5);
}
.btn-primary:active:not(:disabled) { transform: translateY(0); }
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-icon-inner { font-size: 1rem; }

.btn-icon {
    padding: 0.55rem 1rem;
    background: rgba(100,116,139,0.18);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text-2);
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
}
.btn-icon:hover { background: rgba(100,116,139,0.3); color: var(--text-1); }

.btn-icon-sm {
    padding: 0.3rem 0.75rem;
    background: rgba(100,116,139,0.12);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-3);
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.btn-icon-sm:hover { color: var(--text-1); background: rgba(100,116,139,0.25); }

.btn-spinner {
    width: 17px; height: 17px;
    border: 2.5px solid rgba(255,255,255,0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.65s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.hidden { display: none !important; }

.progress-steps {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 1.75rem 1.25rem;
    gap: 0;
}
.step {
display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.45rem;
    }
.step-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--text-3);
    transition: all 0.4s ease;
}
.step.active .step-dot {
    background: var(--accent);
    box-shadow: 0 0 10px rgba(79,142,247,0.7);
    animation: pulse-dot 1.2s ease infinite;
}
.step.done .step-dot {
    background: var(--green);
    box-shadow: 0 0 8px rgba(34,197,94,0.5);
    }
@keyframes pulse-dot {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.4); }
}
.step-label { font-size: 0.68rem; color: var(--text-3); font-weight: 500; white-space: nowrap; text-align: center; transition: color 0.3s; }
.step.active .step-label { color: var(--accent); }
.step.done .step-label  { color: var(--green); }
.step-divider {
    flex: 1;
    height: 1px;
    background: var(--border);
    margin: 0 0.6rem;
    margin-bottom: 1.2rem;
    min-width: 20px;
}

.status-msg {
    text-align: center;
    font-size: 0.88rem;
    min-height: 1rem;
    color: var(--text-2);
    transition: color 0.3s;
    padding: 0 1.75rem 1rem;
}
.status-msg.error   { color: var(--red); }
.status-msg.success { color: var(--green); }

.results-section {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    animation: fadeUp 0.5s ease;
}
@keyframes fadeUp {
from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0); }
}

.overview-row {
display: grid;
    grid-template-columns: 260px 1fr;
    gap: 1.25rem;
    align-items: stretch;
}

.overall-card {
    background: var(--surface);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.25rem 1.5rem;
    text-align: center;
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    }
.overall-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.score-ring-wrapper {
    position: relative;
    width: 140px;
    height: 140px;
}
.score-ring { width: 100%; height: 100%; transform: rotate(-90deg); }
.ring-bg   { fill: none; stroke: rgba(255,255,255,0.05); stroke-width: 10; }
.ring-fill {
    fill: none;
    stroke: url(#ringGradient);
    stroke-width: 10;
    stroke-linecap: round;
     transition: stroke-dashoffset 1.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.ring-score-label {
    position: absolute;
    inset: 0;
     display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.ring-score {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.ring-max { font-size: 0.9rem; color: var(--text-3); font-weight: 500; }
.overall-grade {
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.4rem 1.2rem;
    border-radius: 100px;
    display: inline-block;
}

.radar-card {
    background: var(--surface);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
     box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    }
.radar-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    align-self: flex-start;
}
.radar-wrapper {
width: 100%;
    max-width: 300px;
    aspect-ratio: 1;
}
.radar-svg { width: 100%; height: 100%; overflow: visible; }

.dimensions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 1rem;
    }

.dim-card {
    background: var(--surface);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.35rem;
    box-shadow: var(--shadow);
    ransition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    animation: fadeUp 0.5s ease backwards;
    cursor: default;
}
.dim-card:hover {
transform: translateY(-3px);
    box-shadow: 0 14px 45px rgba(0,0,0,0.5);
    border-color: var(--border-2);
}

.dim-icon  { font-size: 1.5rem; margin-bottom: 0.45rem; }
.dim-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--text-2);
    margin-bottom: 0.75rem;
}
.dim-score { font-size: 1.9rem; font-weight: 800; margin: 0.5rem 0 0.7rem; }

.dim-bar-track {
    height: 5px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
     overflow: hidden;
    margin-bottom: 0.5rem;
}
.dim-bar {
    height: 100%;
    width: 0%;
    border-radius: 3px;
    transition: width 1.1s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.dim-feedback {
font-size: 0.76rem;
    color: var(--text-2);
    line-height: 1.5;
    margin-top: 0.4rem;
}

.score-excellent .dim-score { color: #22c55e; }
.score-excellent .dim-bar   { background: linear-gradient(90deg, #16a34a, #22c55e); }
.score-excellent             { border-color: rgba(34,197,94,0.18); }

.score-good .dim-score { color: #3b82f6; }
.score-good .dim-bar   { background: linear-gradient(90deg, #2563eb, #3b82f6); }
.score-good             { border-color: rgba(59,130,246,0.18); }

.score-average .dim-score { color: #eab308; }
.score-average .dim-bar   { background: linear-gradient(90deg, #ca8a04, #eab308); }
.score-average             { border-color: rgba(234,179,8,0.18); }

.score-poor .dim-score { color: #ef4444; }
.score-poor .dim-bar   { background: linear-gradient(90deg, #dc2626, #ef4444); }
.score-poor             { border-color: rgba(239,68,68,0.18); }

.extracted-card {
    background: var(--surface);
    backdrop-filter: blur(20px);
     border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.65rem;
    box-shadow: var(--shadow);
}
.extracted-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.85rem;
    }
.card-heading {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--text-2);
}
.extracted-body {
    font-family: 'Fira Code', 'JetBrains Mono', monospace;
    font-size: 0.875rem;
    line-height: 1.75;
    color: #cbd5e1;
    background: rgba(0,0,0,0.28);
    padding: 1.2rem;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.04);
    max-height: 280px;
    overflow-y: auto;
    white-space: pre-wrap;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.1) transparent;
}

.suggestions-card {
    background: var(--surface);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.65rem;
    box-shadow: var(--shadow);
}
.suggestions-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
    margin-top: 0.65rem;
}
.suggestion-item {
display: flex;
    gap: 1rem;
    align-items: flex-start;
    background: var(--surface-2);
    border: 1px solid var(--border);
     border-radius: var(--radius-sm);
    padding: 1rem 1.2rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeUp 0.4s ease backwards;
}
.suggestion-item:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 22px rgba(0,0,0,0.3);
}
.suggestion-icon { font-size: 1.35rem; flex-shrink: 0; line-height: 1; }
.suggestion-body  { flex: 1; }
.suggestion-header {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.3rem;
    flex-wrap: wrap;
}
.suggestion-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-1);
}
.suggestion-badge {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 0.18rem 0.55rem;
    border-radius: 100px;
    }
.badge-high   { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.28); }
.badge-medium { background: rgba(234,179,8,0.12);  color: #facc15; border: 1px solid rgba(234,179,8,0.28); }
.badge-low    { background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(59,130,246,0.28); }
.suggestion-cat {
font-size: 0.65rem;
    color: var(--text-3);
    font-weight: 500;
}
.suggestion-detail {
font-size: 0.82rem;
    color: var(--text-2);
    line-height: 1.55;
}

.history-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    margin-top: 2rem;
}
.history-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.stats-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    justify-content: center;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 1rem;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
.stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    min-width: 80px;
    border-right: 1px solid var(--border);
}
.stat-item:last-child {
    border-right: none;
}
.stat-val {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--primary);
}
.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}
