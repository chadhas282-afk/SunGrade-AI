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

.paragraphs-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 1rem;
    padding: 1.5rem;
    margin-bottom: 2rem;
}
.table-responsive {
    overflow-x: auto;
}
.para-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    text-align: left;
    }
.para-table th {
    background: rgba(255, 255, 255, 0.05);
    padding: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
}
.para-table td {
    padding: 0.75rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    vertical-align: middle;
}
.para-table tr:last-child td {
    border-bottom: none;
    }
.para-table .score-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.85rem;
}
.para-table .para-preview {
    color: var(--text-muted);
    font-size: 0.85rem;
    font-style: italic;
    max-width: 300px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.header-actions {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.toggle-switch {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
}
.toggle-switch input {
    display: none;
}
.toggle-slider {
    position: relative;
    width: 36px;
    height: 20px;
    background-color: var(--bg-surface);
    border-radius: 20px;
    border: 1px solid var(--border);
    transition: 0.3s;
    }
.toggle-slider:before {
    content: "";
    position: absolute;
    height: 14px;
     width: 14px;
    left: 2px;
    bottom: 2px;
    background-color: var(--text-muted);
    border-radius: 50%;
     transition: 0.3s;
}
.toggle-switch input:checked + .toggle-slider {
    background-color: var(--primary);
    border-color: var(--primary);
    }
.toggle-switch input:checked + .toggle-slider:before {
    transform: translateX(16px);
    background-color: #fff;
}
.toggle-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    user-select: none;
}
.toggle-switch input:checked ~ .toggle-label {
    color: var(--text);
}

.history-item {
display: flex;
    align-items: center;
    gap: 1rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1.1rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.history-item:hover { border-color: var(--border-2); transform: translateX(4px); }
.history-score-badge {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    font-weight: 800;
    flex-shrink: 0;
}
.history-meta { flex: 1; }
.history-time { font-size: 0.72rem; color: var(--text-3); }
.history-preview { font-size: 0.82rem; color: var(--text-2); margin-top: 0.2rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 400px; }
.history-empty { color: var(--text-3); font-size: 0.85rem; text-align: center; padding: 1rem 0; }

@media (max-width: 900px) {
    .app-wrapper { gap: 1.25rem; }
    .overall-card { padding: 1.5rem 1rem; }
}

@media (max-width: 768px) {
    body { padding: 1rem 0.5rem 3rem; }
    .overview-row { grid-template-columns: 1fr; }
    .dimensions-grid { grid-template-columns: 1fr 1fr; }
    .radar-wrapper { max-width: 320px; margin: 0 auto; }
    .upload-section { padding: 0; }
    .tab-btn { padding: 0.8rem 0.5rem; font-size: 0.85rem; }
    .history-preview { max-width: calc(100vw - 120px); }
    .score-circle { width: 120px; height: 120px; }
    .score-number { font-size: 2.5rem; }
}

@media (max-width: 480px) {
.dimensions-grid { grid-template-columns: 1fr; }
    .progress-steps { gap: 0; flex-direction: column; align-items: flex-start; padding-left: 1rem; }
    .step { margin-bottom: 0.5rem; }
    .step-label { font-size: 0.75rem; }
    .app-title { font-size: 2rem; }
    .tab-btn { flex-direction: column; gap: 0.2rem; font-size: 0.75rem; }
    .btn { width: 100%; justify-content: center; }
    .metrics-row { flex-direction: column; align-items: stretch; }
    .history-meta { flex-direction: column; align-items: flex-start; }
}

.annotated-body {
    font-size: 0.92rem;
    line-height: 1.85;
    color: var(--text-1);
    white-space: pre-wrap;
    word-break: break-word;
    padding: 0.25rem 0;
}

.ann {
    border-radius: 3px;
    padding: 1px 2px;
    cursor: help;
    position: relative;
    transition: opacity 0.15s;
}
.ann:hover { opacity: 0.85; }

.ann-spelling {
background: rgba(239, 68, 68, 0.18);
    border-bottom: 2px dotted #ef4444;
    color: #fca5a5;
}

.ann-transition {
    background: rgba(34, 197, 94, 0.15);
    border-bottom: 2px solid rgba(34,197,94,0.5);
    color: #86efac;
}

.ann-academic {
    background: rgba(139, 92, 246, 0.18);
    border-bottom: 2px solid rgba(139,92,246,0.45);
    color: #c4b5fd;
}

.ann-passive {
    background: rgba(245, 158, 11, 0.15);
    border-bottom: 2px solid rgba(245,158,11,0.5);
    color: #fcd34d;
}

.ann-overused {
    background: rgba(249, 115, 22, 0.15);
    border-bottom: 2px dotted rgba(249,115,22,0.7);
    color: #fdba74;
}

.ann-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem 1.2rem;
    margin-bottom: 1rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
}
.ann-legend-item {
display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: var(--text-2);
}
.ann-swatch {
    display: inline-block;
    width: 14px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
}
.ann-swatch.ann-spelling  { background: rgba(239,68,68,0.5);   border-bottom: 2px dotted #ef4444; }
.ann-swatch.ann-transition { background: rgba(34,197,94,0.4);  border-bottom: 2px solid #22c55e; }
.ann-swatch.ann-academic   { background: rgba(139,92,246,0.4); border-bottom: 2px solid #8b5cf6; }
.ann-swatch.ann-passive    { background: rgba(245,158,11,0.4); border-bottom: 2px solid #f59e0b; }
.ann-swatch.ann-overused   { background: rgba(249,115,22,0.4); border-bottom: 2px dotted #f97316; }

.ann[data-tip]:hover::after {
content: attr(data-tip);
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    background: #1e293b;
    color: #f1f5f9;
    font-size: 0.72rem;
    padding: 4px 8px;
    border-radius: 6px;
     white-space: nowrap;
    pointer-events: none;
    border: 1px solid rgba(255,255,255,0.1);
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}

.dim-card {
    cursor: pointer;
    user-select: none;
    }
.dim-card:focus-visible { outline: 2px solid var(--accent); }

.dim-detail {
    display: none;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
    font-size: 0.78rem;
    color: var(--text-2);
    line-height: 1.6;
    animation: fadeInDown 0.2s ease;
}
.dim-card.expanded .dim-detail { display: block; }

.dim-detail dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.25rem 0.75rem;
}
.dim-detail dt {
    color: var(--text-3);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.dim-detail dd {
    color: var(--text-2);
}

.dim-expand-icon {
    margin-left: auto;
    font-size: 0.7rem;
    color: var(--text-3);
    transition: transform 0.2s ease;
    flex-shrink: 0;
}
.dim-card.expanded .dim-expand-icon { transform: rotate(180deg); }

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
}

</style>
</head>
<body>
    <div class="bg-orbs">
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
    </div>

    <main class="app-wrapper">
        
        <header class="app-header">
            <div class="logo">
                <img src="/static/logo.png" class="logo-icon" alt="Logo" style="height: 18px; width: 18px; border-radius: 4px; object-fit: cover;">
                <span class="logo-text">SunGrade AI</span>
                <span class="logo-version">v4.0</span>
            </div>
            <h1 class="app-title">SunGrade AI</h1>
            <p class="app-subtitle">8-dimensional AI writing assessment — now with Academic Word List scoring, thesis detection &amp; text highlighting</p>
        </header>

        <section class="upload-section" id="inputSection">
            
            <div class="tab-bar" role="tablist" aria-label="Input method">
                <button class="tab-btn active" id="tab-upload" role="tab" aria-selected="true" aria-controls="panel-upload">
                    <span class="tab-icon">📷</span> Upload Photo
                    </button>
                <button class="tab-btn" id="tab-text" role="tab" aria-selected="false" aria-controls="panel-text">
                    <span class="tab-icon">✍️</span> Paste Text
                </button>
            </div>

            <div id="panel-upload" class="tab-panel" role="tabpanel" aria-labelledby="tab-upload">
                <div id="dropzone" class="dropzone" role="button" tabindex="0" aria-label="Upload essay image">
                    <div class="dz-icon">📄</div>
                     <p class="dz-title">Drop your essay photo here</p>
                    <p class="dz-subtitle">or click to browse — JPG, PNG supported</p>
                    <input type="file" id="fileInput" accept=".jpg,.jpeg,.png" hidden aria-label="File input">
                </div>
                <div id="previewContainer" class="preview-container hidden">
                    <img id="imagePreview" src="" alt="Uploaded essay preview">
                    <button id="btnRemove" class="btn-icon" aria-label="Remove image">✕ Remove</button>
                </div>
            </div>

             <div id="panel-text" class="tab-panel hidden" role="tabpanel" aria-labelledby="tab-text">
                <div class="text-input-wrapper">
                    <textarea
                        id="essayTextarea"
                        class="essay-textarea"
                        placeholder="Paste or type your essay here…&#10;&#10;A minimum of 50 words is recommended for a meaningful analysis."
                        aria-label="Essay text input"
                        rows="10"
                        maxlength="50000"
                    ></textarea>
                    <div class="textarea-footer">
                        <span id="charCount" class="char-count">0 words</span>
                        <button id="btnClearText" class="btn-icon-sm">Clear</button>
                    </div>
                </div>
            </div>

            <div class="action-bar">
                <button id="btnAnalyze" class="btn-primary" disabled>
                    <span class="btn-icon-inner">🔬</span>
                    <span class="btn-text">Analyse Essay</span>
                    <span class="btn-spinner hidden" aria-label="Analysing..."></span>
                </button>
            </div>

            <div id="progressSteps" class="progress-steps hidden" role="status" aria-live="polite">
                <div class="step" id="step-ocr">
                    <span class="step-dot"></span>
                    <span class="step-label">Extracting Text (OCR)</span>
                </div>
                <div class="step-divider"></div>
                <div class="step" id="step-nlp">
                    <span class="step-dot"></span>
                    <span class="step-label">NLP Analysis</span>
                </div>
                <div class="step-divider"></div>
                <div class="step" id="step-ml">
                    <span class="step-dot"></span>
                    <span class="step-label">Neural Scoring</span>
                </div>
                <div class="step-divider"></div>
                <div class="step" id="step-done">
                    <span class="step-dot"></span>
                    <span class="step-label">Complete</span>
                </div>
            </div>

            <p id="statusMessage" class="status-msg" role="status" aria-live="polite"></p>
        </section>

        <section id="resultsSection" class="results-section hidden">

            <div class="overview-row">
                
                <div class="overall-card">
                    <div class="overall-label">Overall Score</div>
                    <div class="score-ring-wrapper">
                        <svg class="score-ring" viewBox="0 0 120 120" aria-hidden="true">
                            <defs>
                                <linearGradient id="ringGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%"   stop-color="#60a5fa"/>
                                    <stop offset="50%"  stop-color="#a78bfa"/>
                                    <stop offset="100%" stop-color="#34d399"/>
                                </linearGradient>
                            </defs>
                            <circle class="ring-bg" cx="60" cy="60" r="50"/>
                            <circle id="overallRing" class="ring-fill" cx="60" cy="60" r="50"
                                stroke-dasharray="314" stroke-dashoffset="314"/>
                        </svg>
                        <div class="ring-score-label">
                         <span id="overallScoreNum" class="ring-score">--</span>
                            <span class="ring-max">/ 10</span>
                        </div>
                    </div>
                    <div id="overallGrade" class="overall-grade"></div>
                </div>

                <div class="radar-card">
                    <div class="radar-title">Dimension Radar</div>
                    <div class="radar-wrapper">
                    <svg id="radarChart" class="radar-svg" viewBox="0 0 300 300" aria-label="Radar chart of dimension scores"></svg>
                    </div>
                </div>
            </div>

            <div class="stats-strip" id="statsStrip">
                <div class="stat-item"><span class="stat-val" id="statWords">--</span><span class="stat-label">Words</span></div>
                <div class="stat-item"><span class="stat-val" id="statSents">--</span><span class="stat-label">Sentences</span></div>
                <div class="stat-item"><span class="stat-val" id="statParas">--</span><span class="stat-label">Paragraphs</span></div>
                <div class="stat-item"><span class="stat-val" id="statTime">--</span><span class="stat-label">Min read</span></div>
                <div class="stat-item"><span class="stat-val" id="statGrade">--</span><span class="stat-label">Grade Lvl</span></div>
                <div class="stat-item"><span class="stat-val" id="statSpell">--</span><span class="stat-label">Errors</span></div>
            </div>

            <div class="dimensions-grid" id="dimensionsGrid">
                
            </div>

            <div class="suggestions-card">
                <h2 class="card-heading">💬 Improvement Suggestions</h2>
                <ul id="suggestionsList" class="suggestions-list"></ul>
            </div>

            <div class="paragraphs-card">
                <h2 class="card-heading">📊 Paragraph Breakdown</h2>
                <div class="table-responsive">
                    <table class="para-table">
                        <thead>
                            <tr>
                                <th>¶</th>
                                <th>Role</th>
                                <th>Words</th>
                                <th>Avg Len</th>
                                <th>Score</th>
                                <th>Preview</th>
                            </tr>
                        </thead>
                        <tbody id="paragraphsTableBody">
                            
                        </tbody>
                         </table>
                </div>
            </div>

            <div class="extracted-card">
                <div class="extracted-header">
                    <h2 class="card-heading">📄 Analysed Text</h2>
                    <div class="header-actions">
                        <label class="toggle-switch">
                            <input type="checkbox" id="toggleAnnotations" checked>
                            <span class="toggle-slider"></span>
                            <span class="toggle-label">Highlights</span>
                        </label>
                        <button id="btnCopyText" class="btn-icon-sm">Copy</button>
                    </div>
                </div>
                
                <div class="ann-legend" id="annLegend">
                    <span class="ann-legend-item">
                        <span class="ann-swatch ann-spelling"></span>Spelling error
                    </span>
                    <span class="ann-legend-item">
                        <span class="ann-swatch ann-transition"></span>Transition word
                    </span>
                    <span class="ann-legend-item">
                        <span class="ann-swatch ann-academic"></span>Academic word (AWL)
                    </span>
                    <span class="ann-legend-item">
                        <span class="ann-swatch ann-passive"></span>Passive voice
                    </span>
                    <span class="ann-legend-item">
                        <span class="ann-swatch ann-overused"></span>Overused word
                    </span>
                </div>
                <div id="annotatedText" class="annotated-body"></div>
                
                <div id="extractedText" class="extracted-body" style="display:none"></div>
            </div>

            <div class="history-card" id="historyCard">
            <h2 class="card-heading">🕐 Recent Analyses</h2>
                <div id="historyList" class="history-list"></div>
            </div>
        </section>
    </main>

    <script>

const tabUpload      = document.getElementById('tab-upload');
const tabText        = document.getElementById('tab-text');
const panelUpload    = document.getElementById('panel-upload');
const panelText      = document.getElementById('panel-text');
const dropzone       = document.getElementById('dropzone');
const fileInput      = document.getElementById('fileInput');
const prevCont       = document.getElementById('previewContainer');
const imgPreview     = document.getElementById('imagePreview');
const btnRemove      = document.getElementById('btnRemove');
const essayTextarea  = document.getElementById('essayTextarea');
const charCount      = document.getElementById('charCount');
const btnClearText   = document.getElementById('btnClearText');
const btnAnalyze     = document.getElementById('btnAnalyze');
const btnText        = btnAnalyze.querySelector('.btn-text');
const btnSpinner     = btnAnalyze.querySelector('.btn-spinner');
const statusMsg      = document.getElementById('statusMessage');
const resultsSection = document.getElementById('resultsSection');
const progressSteps  = document.getElementById('progressSteps');
const dimensionsGrid = document.getElementById('dimensionsGrid');
const historyCard    = document.getElementById('historyCard');
const historyList    = document.getElementById('historyList');
const btnCopyText    = document.getElementById('btnCopyText');

let selectedFile  = null;
let activeTab     = 'upload';
const HISTORY_KEY = 'sungrade_history';
const MAX_HISTORY = 3;

const DIM_ORDER = [
    { key: 'readability',   label: 'Readability',   icon: '📖' },
    { key: 'vocabulary',    label: 'Vocabulary',     icon: '🔤' },
    { key: 'structure',     label: 'Structure',      icon: '📐' },
    { key: 'spelling',      label: 'Spelling',       icon: '✅' },
    { key: 'transitions',   label: 'Transitions',    icon: '🔗' },
    { key: 'passive_voice', label: 'Active Voice',   icon: '💪' },
    { key: 'argument',      label: 'Argument',       icon: '🎯' },
    { key: 'coherence',     label: 'Coherence',      icon: '🧠' },
];

function switchTab(tab) {
    activeTab = tab;
    tabUpload.classList.toggle('active', tab === 'upload');
    tabText.classList.toggle('active', tab === 'text');
    tabUpload.setAttribute('aria-selected', tab === 'upload');
    tabText.setAttribute('aria-selected', tab === 'text');
    panelUpload.classList.toggle('hidden', tab !== 'upload');
    panelText.classList.toggle('hidden', tab !== 'text');
    updateAnalyzeBtn();
    setStatus('');
}

tabUpload.addEventListener('click', () => switchTab('upload'));
tabText.addEventListener('click',   () => switchTab('text'));

dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

function handleFile(file) {
    const valid = ['image/jpeg', 'image/png'];
    if (!valid.includes(file.type)) {
        setStatus('Please upload a valid JPG or PNG image.', 'error');
        return;
    }
    selectedFile = file;
    setStatus('');
    const reader = new FileReader();
    reader.onload = e => {
        imgPreview.src = e.target.result;
        dropzone.classList.add('hidden');
        prevCont.classList.remove('hidden');
        updateAnalyzeBtn();
    };
    reader.readAsDataURL(file);
}

btnRemove.addEventListener('click', resetUpload);
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    imgPreview.src = '';
    prevCont.classList.add('hidden');
    dropzone.classList.remove('hidden');
    updateAnalyzeBtn();
    setStatus('');
    }

essayTextarea.addEventListener('input', () => {
    const wordCount = countWords(essayTextarea.value);
    charCount.textContent = `${wordCount} word${wordCount !== 1 ? 's' : ''}`;
    updateAnalyzeBtn();
});

btnClearText.addEventListener('click', () => {
    essayTextarea.value = '';
    charCount.textContent = '0 words';
    updateAnalyzeBtn();
});

function countWords(text) {
    return (text.trim().match(/\b\w+\b/g) || []).length;
}

function updateAnalyzeBtn() {
    if (activeTab === 'upload') {
    btnAnalyze.disabled = !selectedFile;
    } else {
        btnAnalyze.disabled = countWords(essayTextarea.value) < 5;
    }
}

const STEPS = ['step-ocr', 'step-nlp', 'step-ml', 'step-done'];

function resetSteps() {
    STEPS.forEach(id => {
    const el = document.getElementById(id);
        el.classList.remove('active', 'done');
    });
}

function advanceStep(index) {
    STEPS.forEach((id, i) => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'done');
        if (i < index) el.classList.add('done');
        if (i === index) el.classList.add('active');
    });
}

btnAnalyze.addEventListener('click', async () => {
    btnText.textContent = 'Analysing…';
    btnSpinner.classList.remove('hidden');
    btnAnalyze.disabled = true;
    resultsSection.classList.add('hidden');
    progressSteps.classList.remove('hidden');
    resetSteps();

    if (activeTab === 'upload') {
        await analyzeImage();
    } else {
        await analyzeText();
    }

    btnText.textContent = 'Analyse Essay';
    btnSpinner.classList.add('hidden');
    updateAnalyzeBtn();
});

async function analyzeImage() {
    if (!selectedFile) return;
    if (btnRemove) btnRemove.disabled = true;

    advanceStep(0);
    setStatus('Running OCR — extracting text from image…');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {

        await delay(600);
        advanceStep(1);
        setStatus('Running NLP analysis across 8 dimensions…');

        const response = await fetch('/api/analyze', { method: 'POST', body: formData });
        const data = await response.json();

        advanceStep(2);
        setStatus('Neural scoring complete…');
        await delay(400);
        advanceStep(3);

        if (response.ok && data.status === 'success') {
            await delay(300);
            renderResults(data);
            setStatus('Analysis complete!', 'success');
        } else {
            setStatus(data.error || 'An error occurred.', 'error');
            progressSteps.classList.add('hidden');
        }
    } catch (err) {
        console.error(err);
        setStatus('Network error. Please try again.', 'error');
        progressSteps.classList.add('hidden');
    } finally {
     if (btnRemove) btnRemove.disabled = false;
    }
}

async function analyzeText() {
    const text = essayTextarea.value.trim();
    if (!text) return;

    advanceStep(1);
    setStatus('Running NLP analysis across 8 dimensions…');

    try {
        await delay(300);
        advanceStep(2);

        const response = await fetch('/api/analyze-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await response.json();

        setStatus('Neural scoring complete…');
        await delay(400);
        advanceStep(3);

        if (response.ok && data.status === 'success') {
            await delay(250);
            renderResults(data);
            setStatus('Analysis complete!', 'success');
            } else {
            setStatus(data.error || 'An error occurred.', 'error');
            progressSteps.classList.add('hidden');
        }
    } catch (err) {
        console.error(err);
        setStatus('Network error. Please try again.', 'error');
        progressSteps.classList.add('hidden');
    }
}

function renderResults(data) {
    const dims = data.dimensions;
    const overall = data.overall_score;

    document.getElementById('overallScoreNum').textContent = overall.toFixed(1);
    const circumference = 314;
    const offset = circumference * (1 - overall / 10);
    const ring = document.getElementById('overallRing');
    ring.setAttribute('stroke-dashoffset', circumference);
    setTimeout(() => ring.setAttribute('stroke-dashoffset', offset), 50);

    const gradeEl = document.getElementById('overallGrade');
    const { label, color } = getGrade(overall);
    gradeEl.textContent = label;
    gradeEl.style.background = color + '22';
    gradeEl.style.color = color;
    gradeEl.style.border = `1px solid ${color}55`;

    dimensionsGrid.innerHTML = '';
    DIM_ORDER.forEach(({ key, label: dimLabel, icon }, idx) => {
        const dim = dims[key];
        if (!dim) return;
        const score = dim.score;
        const tier  = getTierClass(score);

        let detailHtml = '';
        if (dim.detail && Object.keys(dim.detail).length > 0) {
            const rows = [];
            const d = dim.detail;
            if (key === 'vocabulary') {
                rows.push(`<dt>AWL words</dt><dd>${d.awl_count || 0} (${d.awl_pct || 0}% of content words)</dd>`);
            } else if (key === 'structure') {
                rows.push(`<dt>Thesis detected</dt><dd>${d.has_thesis ? '✅ Yes' : '❌ No'}</dd>`);
                rows.push(`<dt>Conclusion detected</dt><dd>${d.has_conclusion ? '✅ Yes' : '❌ No'}</dd>`);
            } else if (key === 'spelling' && d.errors && d.errors.length > 0) {
                rows.push(`<dt>Errors found</dt><dd>${d.errors.slice(0,8).map(escapeHtml).join(', ')}</dd>`);
            } else if (key === 'transitions' && d.found && d.found.length > 0) {
                rows.push(`<dt>Found</dt><dd>${d.found.slice(0,8).map(escapeHtml).join(', ')}</dd>`);
            }
            if (rows.length) {
                detailHtml = `<div class="dim-detail"><dl>${rows.join('')}</dl></div>`;
            }
        }

        const card = document.createElement('div');
        card.className = `dim-card ${tier}`;
        card.id = `card-${key}`;
        card.style.animationDelay = `${idx * 0.06}s`;
        card.setAttribute('tabindex', '0');
        card.setAttribute('role', 'button');
        card.setAttribute('aria-expanded', 'false');

        card.innerHTML = `
            <div class="dim-icon">${dim.icon || icon}</div>
            <div class="dim-label">${dim.label || dimLabel}</div>
            <div class="dim-bar-track"><div class="dim-bar" id="bar-${key}"></div></div>
            <div class="dim-score">${score.toFixed(1)}</div>
            <div class="dim-feedback">${escapeHtml(dim.feedback)}</div>
            ${detailHtml}
            ${detailHtml ? '<div class="dim-expand-icon">▼</div>' : ''}
        `;
        dimensionsGrid.appendChild(card);

        if (detailHtml) {
            card.addEventListener('click', () => {
                const expanded = card.classList.toggle('expanded');
                card.setAttribute('aria-expanded', expanded);
            });
            card.addEventListener('keydown', e => {
            if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    card.click();
                }
            });
        }

        setTimeout(() => {
            const bar = document.getElementById(`bar-${key}`);
            if (bar) bar.style.width = `${score * 10}%`;
            }, 100 + idx * 60);
    });

    renderRadar(dims);

    if (data.stats) {
        document.getElementById('statWords').textContent = data.stats.word_count || 0;
        document.getElementById('statSents').textContent = data.stats.sentence_count || 0;
        document.getElementById('statParas').textContent = data.stats.paragraph_count || 0;
        document.getElementById('statTime').textContent  = data.stats.reading_time_min || 0;
        document.getElementById('statGrade').textContent = data.stats.grade_level || 0;
        document.getElementById('statSpell').textContent = data.stats.spelling_errors || 0;
    }

    if (data.paragraphs) {
        renderParagraphs(data.paragraphs);
    }

    const rawText = data.extracted_text || '';
    document.getElementById('extractedText').textContent = rawText;
    renderAnnotatedText(rawText, data.annotations || []);

    renderSuggestions(data.suggestions || []);

    saveHistory(data);
    renderHistory();

    resultsSection.classList.remove('hidden');
    setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

function renderRadar(dims) {
    const svg = document.getElementById('radarChart');
    svg.innerHTML = '';

    const cx = 150, cy = 150, maxR = 108;
    const keys = DIM_ORDER.map(d => d.key);
    const labels = DIM_ORDER.map(d => d.label);
    const n = keys.length;
    const scores = keys.map(k => (dims[k] ? dims[k].score : 0) / 10);

    function polar(angle, radius) {
        const a = (angle - Math.PI / 2);
        return [cx + radius * Math.cos(a), cy + radius * Math.sin(a)];
    }

    const angleStep = (2 * Math.PI) / n;

    const gridLevels = [0.2, 0.4, 0.6, 0.8, 1.0];
    gridLevels.forEach(level => {
    const pts = keys.map((_, i) => polar(i * angleStep, maxR * level));
        const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        poly.setAttribute('points', pts.map(p => p.join(',')).join(' '));
        poly.setAttribute('fill', 'none');
        poly.setAttribute('stroke', 'rgba(255,255,255,0.06)');
        poly.setAttribute('stroke-width', '1');
        svg.appendChild(poly);
    });

    keys.forEach((_, i) => {
    const [x, y] = polar(i * angleStep, maxR);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', cx); line.setAttribute('y1', cy);
        line.setAttribute('x2', x);  line.setAttribute('y2', y);
        line.setAttribute('stroke', 'rgba(255,255,255,0.08)');
        line.setAttribute('stroke-width', '1');
        svg.appendChild(line);
    });

    const scorePts = scores.map((s, i) => polar(i * angleStep, maxR * s));
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', scorePts.map(p => p.join(',')).join(' '));
    poly.setAttribute('fill', 'rgba(79,142,247,0.18)');
    poly.setAttribute('stroke', 'rgba(79,142,247,0.75)');
    poly.setAttribute('stroke-width', '2');
    poly.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(poly);

    scorePts.forEach(([x, y], i) => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x); circle.setAttribute('cy', y); circle.setAttribute('r', '4');
        circle.setAttribute('fill', getScoreColor(scores[i]));
        circle.setAttribute('stroke', 'rgba(255,255,255,0.5)');
        circle.setAttribute('stroke-width', '1.2');
        svg.appendChild(circle);
    });

    for (let i = 0; i < n; i++) {
        const a = (Math.PI * 2 * i) / n - Math.PI / 2;
        const x = cx + Math.cos(a) * (maxR + 25);
        const y = cy + Math.sin(a) * (maxR + 25);
        svg.appendChild(createSvg('text', {
            x, y,
            'text-anchor': 'middle',
            'dominant-baseline': 'middle',
            class: 'radar-label'
        }, labels[i]));
    }
}

function createSvg(tag, attrs, text) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    if (text) el.textContent = text;
    return el;
}

function getScoreColor(normalizedScore) {
    if (normalizedScore >= 0.75) return '#22c55e';
    if (normalizedScore >= 0.55) return '#3b82f6';
    if (normalizedScore >= 0.35) return '#eab308';
    return '#ef4444';
}

function renderParagraphs(paragraphs) {
    const tbody = document.getElementById('paragraphsTableBody');
    tbody.innerHTML = '';

    if (!paragraphs || paragraphs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6">No paragraph data available.</td></tr>';
        return;
    }

    paragraphs.forEach(p => {
        const tr = document.createElement('tr');
        
        let colorClass = 'score-high';
        if (p.score < 5.0) colorClass = 'score-low';
        else if (p.score < 7.5) colorClass = 'score-mid';

         tr.innerHTML = `
            <td><strong>${p.index}</strong></td>
            <td><span style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">${p.role}</span></td>
            <td>${p.words}</td>
            <td>${p.avg_len}</td>
            <td><span class="score-badge ${colorClass}">${p.score.toFixed(1)}</span></td>
            <td class="para-preview">${escapeHtml(p.preview)}</td>
        `;
        tbody.appendChild(tr);
    });
    }

function renderAnnotatedText(text, annotations) {
    const container = document.getElementById('annotatedText');
    container.innerHTML = '';

    if (!text) return;

    const sorted = [...annotations]
        .filter(a => typeof a.start === 'number' && typeof a.end === 'number'
        && a.start >= 0 && a.end > a.start && a.start < text.length)
        .sort((a, b) => a.start - b.start);

    const tips = {
        spelling:   ann => ann.suggestion ? `Spelling error \u2014 try \u201c${ann.suggestion}\u201d` : 'Spelling error',
        transition: () => 'Transition word \u2705',
        academic:   () => 'Academic word (AWL) \u2705',
        passive:    () => 'Passive voice construction',
        overused:   () => 'Overused word',
    };

    let cursor = 0;

    for (const ann of sorted) {

        if (ann.start < cursor) continue;

        const safeEnd = Math.min(ann.end, text.length);
        if (safeEnd <= ann.start) continue;

        if (ann.start > cursor) {
            container.appendChild(document.createTextNode(text.slice(cursor, ann.start)));
        }

        const span = document.createElement('span');
        span.className = `ann ann-${ann.tag}`;
        span.textContent = text.slice(ann.start, safeEnd);
        const tipFn = tips[ann.tag];
        if (tipFn) span.setAttribute('data-tip', tipFn(ann));
        container.appendChild(span);

        cursor = safeEnd;
    }

    if (cursor < text.length) {
        container.appendChild(document.createTextNode(text.slice(cursor)));
    }

    if (!container.hasChildNodes()) {
        container.textContent = text;
         }
}

function renderSuggestions(suggestions) {
    const list = document.getElementById('suggestionsList');
    list.innerHTML = '';

    if (!suggestions || suggestions.length === 0) {
        const li = document.createElement('li');
        li.className = 'suggestion-item';
        li.textContent = 'Great work! No major suggestions at this time.';
        list.appendChild(li);
        return;
    }

    suggestions.forEach((s, i) => {
        const li = document.createElement('li');
        li.className = 'suggestion-item';
        li.style.animationDelay = `${i * 0.065}s`;

        const icon = document.createElement('div');
        icon.className = 'suggestion-icon';
        icon.textContent = s.icon || '💡';

        const body = document.createElement('div');
        body.className = 'suggestion-body';

        const header = document.createElement('div');
        header.className = 'suggestion-header';

        const title = document.createElement('span');
        title.className = 'suggestion-title';
        title.textContent = s.title;

        const badge = document.createElement('span');
        badge.className = `suggestion-badge badge-${s.priority}`;
        badge.textContent = s.priority;

        const cat = document.createElement('span');
        cat.className = 'suggestion-cat';
        cat.textContent = s.category;

        header.appendChild(title);
        header.appendChild(badge);
        header.appendChild(cat);

        const detail = document.createElement('p');
        detail.className = 'suggestion-detail';
        detail.textContent = s.detail;

        body.appendChild(header);
        body.appendChild(detail);
        li.appendChild(icon);
        li.appendChild(body);
        list.appendChild(li);
    });
}

function saveHistory(data) {
    const history = getHistory();
    const entry = {
        timestamp: Date.now(),
        overall: data.overall_score,
        preview: (data.extracted_text || '').slice(0, 80),
        mode: data.mode || 'ocr',
    };
    history.unshift(entry);
    const trimmed = history.slice(0, MAX_HISTORY);
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed)); } catch {}
}

function getHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; }
}

function renderHistory() {
    const history = getHistory();
    historyList.innerHTML = '';

    if (history.length === 0) {
    const empty = document.createElement('div');
        empty.className = 'history-empty';
        empty.textContent = 'No previous analyses yet.';
        historyList.appendChild(empty);
        return;
    }

    history.forEach(entry => {
        const item = document.createElement('div');
        item.className = 'history-item';
         const { label, color } = getGrade(entry.overall);
        const timeStr = new Date(entry.timestamp).toLocaleString();

        item.innerHTML = `
            <div class="history-score-badge" style="background:${color}22; color:${color}; border:1.5px solid ${color}55;">
                ${entry.overall.toFixed(1)}
            </div>
            <div class="history-meta">
                <div class="history-time">${timeStr} · ${entry.mode === 'text' ? '✍️ Text' : '📷 Image'} · ${label}</div>
                <div class="history-preview">${escapeHtml(entry.preview)}…</div>
                </div>
        `;
        historyList.appendChild(item);
    });
}

btnCopyText.addEventListener('click', async () => {
    const text = document.getElementById('extractedText').textContent;
    if (!text) return;
    try {
    await navigator.clipboard.writeText(text);
        btnCopyText.textContent = 'Copied!';
        setTimeout(() => { btnCopyText.textContent = 'Copy'; }, 2000);
    } catch {
        btnCopyText.textContent = 'Failed';
        setTimeout(() => { btnCopyText.textContent = 'Copy'; }, 2000);
    }
});

const toggleAnn = document.getElementById('toggleAnnotations');
if (toggleAnn) {
    toggleAnn.addEventListener('change', (e) => {
        const annBody = document.getElementById('annotatedText');
        const extBody = document.getElementById('extractedText');
        const legend = document.getElementById('annLegend');
        
        if (e.target.checked) {
            annBody.style.display = 'block';
            legend.style.display = 'flex';
            extBody.style.display = 'none';
            } else {
            annBody.style.display = 'none';
            legend.style.display = 'none';
            extBody.style.display = 'block';
        }
    });
}

function getGrade(score) {
    if (score >= 8.5) return { label: 'Outstanding ⭐', color: '#22c55e' };
    if (score >= 7.0) return { label: 'Excellent 🏆',   color: '#3b82f6' };
    if (score >= 5.5) return { label: 'Good 👍',          color: '#8b5cf6' };
    if (score >= 4.0) return { label: 'Average 📝',      color: '#eab308' };
    return { label: 'Needs Work 🔧', color: '#ef4444' };
}

function getTierClass(score) {
    if (score >= 7.5) return 'score-excellent';
    if (score >= 5.5) return 'score-good';
    if (score >= 3.5) return 'score-average';
    return 'score-poor';
}

function setStatus(msg, type = '') {
    statusMsg.textContent = msg;
    statusMsg.className = 'status-msg';
    if (type) statusMsg.classList.add(type);
}

function escapeHtml(str) {
    return (str || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

renderHistory();

</script>
</body>
</html>
"""

spell = SpellChecker()

_AWL_WORDS = {

    'analyse', 'analyze', 'approach', 'area', 'assess', 'assume', 'authority',
    'available', 'benefit', 'concept', 'consist', 'constitute', 'context',
    'contract', 'create', 'data', 'define', 'derive', 'distribute', 'economy',
    'environment', 'establish', 'estimate', 'evident', 'export', 'factor',
    'finance', 'formula', 'function', 'identify', 'income', 'indicate',
    'individual', 'interpret', 'involve', 'issue', 'labour', 'legal', 'legislate',
    'major', 'method', 'occur', 'percent', 'period', 'policy', 'principle',
    'procedure', 'process', 'require', 'research', 'response', 'role',
    'section', 'sector', 'significant', 'similar', 'source', 'specific',
    'structure', 'theory', 'vary', 'affect', 'achieve', 'impact',

    'achieve', 'acquisition', 'administration', 'affect', 'appropriate',
    'aspect', 'assistance', 'category', 'chapter', 'commission', 'community',
    'complex', 'compute', 'conclude', 'conduct', 'consequence', 'construct',
    'consume', 'credit', 'culture', 'design', 'distinction', 'element',
    'equate', 'evaluate', 'feature', 'final', 'focus', 'impact', 'injure',
    'instance', 'interact', 'justify', 'layer', 'link', 'locate', 'maximize',
    'mechanism', 'minimize', 'obvious', 'participate', 'perceive', 'positive',
    'potential', 'previous', 'primary', 'purchase', 'range', 'region',
    'regulate', 'relevant', 'reside', 'resources', 'restrict', 'secure',
    'seek', 'select', 'site', 'strategy', 'survey', 'text', 'traditional',
    'transfer',

    'alternative', 'circumstance', 'comment', 'compensate', 'component',
    'consent', 'considerable', 'constant', 'constrain', 'contribute',
    'convene', 'coordinate', 'core', 'corporate', 'correspond', 'criteria',
    'deduce', 'demonstrate', 'document', 'dominate', 'emphasis', 'ensure',
    'exclude', 'framework', 'fund', 'illustrate', 'immigrate', 'imply',
    'initial', 'instance', 'interact', 'justify', 'layer', 'maintain',
    'modify', 'monitor', 'negate', 'normal', 'obtain', 'parallel', 'parameter',
    'phase', 'predict', 'principal', 'proportion', 'publish', 'react',
    'register', 'rely', 'remove', 'scheme', 'sequence', 'significant',
    'similar', 'simulate', 'sufficient', 'task', 'technical', 'technique',
    'technology', 'valid', 'volume',

    'access', 'adequate', 'annual', 'apparent', 'approximate', 'attitude',
    'attribute', 'civil', 'code', 'commit', 'communicate', 'concentrate',
    'confer', 'contrast', 'cycle', 'debate', 'despite', 'dimension',
    'domestic', 'emerge', 'emphasis', 'ensure', 'exhibit', 'exploit',
    'fluctuate', 'guideline', 'hence', 'hypothesis', 'implement', 'impose',
    'integrate', 'internal', 'investigate', 'liberal', 'likewise', 'media',
    'mode', 'network', 'notion', 'objective', 'orient', 'outcome', 'output',
    'overall', 'passive', 'perceive', 'plus', 'potential', 'previous',
    'primary', 'professional', 'prohibit', 'project', 'promote', 'proportion',
    'pursue', 'radical', 'random', 'reinforce', 'restore', 'retain', 'series',
    'statistic', 'status', 'straightforward', 'subsequent', 'substitute',
    'sustain', 'symbolic', 'target', 'transition', 'trend', 'ultimate',
    'underlie', 'utilize',

    'abstract', 'acknowledge', 'aggregate', 'allocate', 'amend', 'analogy',
    'anticipate', 'apparent', 'appendix', 'appreciate', 'arbitrary',
    'assemble', 'coherent', 'coincide', 'collapse', 'compile', 'complement',
    'comprehensive', 'comprise', 'concurrent', 'confine', 'conform', 'converse',
    'convince', 'correlate', 'diminish', 'discrete', 'discriminate', 'displace',
    'display', 'diverse', 'domain', 'draft', 'duration', 'dynamic', 'eliminate',
    'empirical', 'enable', 'encounter', 'enhance', 'enumerate', 'equate',
    'evolve', 'extract', 'facilitate', 'fundamental', 'generate', 'global',
    'grant', 'hierarchy', 'highlight', 'implicit', 'index', 'induce',
    'inevitable', 'infrastructure', 'inherent', 'innovate', 'insight',
    'intermediate', 'intrinsic', 'label', 'legitimate', 'logic', 'marginal',
    'mature', 'mediate', 'mental', 'minimal', 'nuclear', 'offset', 'option',
    'orient', 'paradigm', 'phenomenon', 'priority', 'prospect', 'rational',
    'react', 'recover', 'refine', 'regime', 'reinforce', 'reluctant', 'resolve',
    'revise', 'rigid', 'scope', 'seek', 'shift', 'stable', 'subordinate',
    'sum', 'supplement', 'suspend', 'systematic', 'terminate', 'thesis',
    'trace', 'transform', 'undergo', 'unify', 'unique', 'undertake', 'visible',
    'voluntary', 'widespread',
}

_TRANSITION_WORDS = {
    'furthermore', 'moreover', 'additionally', 'in addition', 'also',
    'besides', 'likewise', 'similarly',
    'however', 'nevertheless', 'on the other hand', 'in contrast',
    'although', 'whereas', 'conversely', 'despite', 'yet',
    'therefore', 'consequently', 'as a result', 'thus', 'hence',
    'accordingly', 'for this reason',
    'firstly', 'secondly', 'thirdly', 'finally', 'subsequently',
    'next', 'then', 'afterwards',
    'indeed', 'in fact', 'certainly', 'clearly', 'notably',
    'in conclusion', 'to summarise', 'to summarize', 'in summary',
    'ultimately', 'overall',
    'for example', 'for instance', 'such as', 'namely',
    }

_ARGUMENT_MARKERS = {
    'because', 'since', 'therefore', 'thus', 'hence', 'consequently',
    'as a result', 'for example', 'for instance', 'evidence shows',
    'research indicates', 'studies suggest', 'according to',
    'this demonstrates', 'this suggests', 'this shows',
    'it is argued', 'it can be argued',
}

_THESIS_PATTERNS = [
    r'\bthis essay (will |aims to |seeks to |argues?|examines?|explores?)',
    r'\bi (argue|contend|believe|maintain|claim)\b',
    r'\bthe (purpose|aim|objective|goal) of this',
    r'\bit (is|can be) (argued|maintained|claimed|suggested)\b',
    r'\bthis (paper|report|essay|work) (will |aims to |seeks to )',
    r'\bthe (main|central|key) (argument|thesis|claim|point)',
]

_CONCLUSION_PATTERNS = [
    r'\bin conclusion\b', r'\bto (summarise|summarize)\b',
    r'\bin summary\b', r'\boverall[,.]', r'\bto conclude\b',
    r'\bfinally[,.]', r'\bultimately[,.]', r'\bthus[,.]',
]

_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'it', 'in', 'on',
    'of', 'to', 'and', 'or', 'but', 'be', 'has', 'have', 'that', 'this',
    'for', 'not', 'with', 'as', 'at', 'by', 'from', 'they', 'we', 'he',
    'she', 'you', 'i', 'do', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'can', 'our', 'their', 'its', 'been', 'had', 'also',
    'than', 'then', 'when', 'if', 'so', 'up', 'out', 'my', 'your', 'his',
    'her', 'them', 'which', 'who', 'what', 'all', 'more', 'about', 'into',
}

def _sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

def _words(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def _normalize(value, min_val, max_val):
    
    if max_val == min_val:
        return 5.0
    normalized = (value - min_val) / (max_val - min_val)
    return round(max(1.0, min(10.0, 1 + normalized * 9)), 1)

def score_readability(text):

    ease = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)

    if grade < 6:
        score = _normalize(grade, 0, 6)
    elif grade <= 14:
        score = _normalize(grade, 6, 14)
    else:
        score = max(1.0, 10.0 - (grade - 14) * 0.5)

    feedback = f"Flesch Reading Ease: {ease:.1f}/100 | Grade Level: {grade:.1f}. "
    if grade < 6:
        feedback += "Writing is too simple. Use more complex sentence structures."
    elif grade <= 12:
        feedback += "Readability is excellent — well-structured sentences."
    else:
        feedback += "Writing may be overly complex. Consider shorter sentences."

    return round(score, 1), feedback

def score_vocabulary(text):
    
    words = _words(text)
    if not words:
        return 1.0, "No words detected.", 0, 0

    total   = len(words)
    unique  = len(set(words))
    cttr    = unique / math.sqrt(2 * total) if total > 0 else 0

    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 3]
    awl_count  = sum(1 for w in content_words if w in _AWL_WORDS)
    awl_ratio  = awl_count / max(len(content_words), 1)

    cttr_score = _normalize(cttr, 2.0, 10.0)

    awl_score  = _normalize(awl_ratio, 0.02, 0.25)

    score = round(cttr_score * 0.55 + awl_score * 0.45, 1)

    feedback = (
        f"Words: {total} | Unique: {unique} | CTTR: {cttr:.2f} | "
        f"Academic words: {awl_count} ({awl_ratio * 100:.1f}% of content words). "
    )
    if awl_ratio < 0.05:
        feedback += "Very few academic words. Use subject-specific vocabulary to strengthen your writing."
    elif awl_ratio < 0.12:
        feedback += "Moderate academic vocabulary. Introducing more AWL words would elevate your essay."
    else:
        feedback += "Strong academic vocabulary — writing feels scholarly and precise."

    return score, feedback, awl_count, awl_ratio

def score_structure(text):
    
    sentences  = [s for s in _sentences(text) if len(s.strip()) > 3]
    words      = _words(text)
    paragraphs = [p.strip() for p in text.strip().split('\n\n') if p.strip()]

    num_sentences  = len(sentences)
    num_words      = len(words)
    num_paragraphs = len(paragraphs)
    avg_sent_len   = num_words / max(num_sentences, 1)

    if num_sentences > 2:
        sent_lens = [len(re.findall(r'\b[a-zA-Z]+\b', s)) for s in sentences]
        variance  = sum((l - avg_sent_len) ** 2 for l in sent_lens) / num_sentences
        std_dev   = math.sqrt(variance)
        variance_bonus = min(std_dev / 15.0, 1.0)
    else:
        variance_bonus = 0.0

    first_sent = sentences[0].lower() if sentences else ''
    has_thesis = any(re.search(p, first_sent) for p in _THESIS_PATTERNS)

    last_para = paragraphs[-1].lower() if paragraphs else ''
    has_conclusion = any(re.search(p, last_para) for p in _CONCLUSION_PATTERNS)

    if avg_sent_len < 8:
        sent_score = _normalize(avg_sent_len, 0, 8)
    elif avg_sent_len <= 25:
        sent_score = _normalize(avg_sent_len, 8, 25)
    else:
        sent_score = max(1.0, 10.0 - (avg_sent_len - 25) * 0.2)

    length_score = _normalize(min(num_words, 500), 30, 500)
    para_score   = _normalize(min(num_paragraphs, 5), 1, 5)

    thesis_bonus     = 1.0 if has_thesis else 0.0
    conclusion_bonus = 1.0 if has_conclusion else 0.0

    score = round(
        sent_score     * 0.40 +
        length_score   * 0.25 +
        para_score     * 0.15 +
        variance_bonus * 10 * 0.10 +
        thesis_bonus   * 0.05 +                           
        conclusion_bonus * 0.05,
        1
    )
    score = max(1.0, min(10.0, score))

    feedback = (
        f"Sentences: {num_sentences} | Words: {num_words} | "
        f"Paragraphs: {num_paragraphs} | Avg sentence: {avg_sent_len:.1f} words. "
    )
    parts = []
    if has_thesis:
        parts.append("✓ Thesis detected")
    if has_conclusion:
         parts.append("✓ Conclusion detected")
    if parts:
        feedback += " | ".join(parts) + ". "
    if avg_sent_len < 8:
        feedback += "Sentences are too short. Develop your ideas more fully."
    elif avg_sent_len <= 25:
        feedback += "Sentence structure is well-balanced."
    else:
        feedback += "Some sentences are very long. Consider breaking them up."

    return round(score, 1), feedback, has_thesis, has_conclusion

def score_spelling(text):
    
    words = _words(text)
    if not words:
        return 1.0, "No words to check.", set()

    misspelled = spell.unknown(words)
    total      = len(words)
    error_rate = len(misspelled) / total

    score = _normalize(1 - error_rate, 0.7, 1.0)
    sample_errors = list(misspelled)[:5]
    feedback = (
        f"Spelling errors: {len(misspelled)}/{total} words "
        f"({error_rate * 100:.1f}% error rate). "
    )
    if error_rate < 0.05:
        feedback += "Excellent spelling accuracy."
        elif error_rate < 0.15:
        feedback += f"Minor issues. Check: {', '.join(sample_errors)}."
    else:
        feedback += f"Many errors detected. Review: {', '.join(sample_errors)}."

    return round(score, 1), feedback, misspelled

def score_transitions(text):
    
    lower = text.lower()
    sentences = [s for s in _sentences(text) if len(s.strip()) > 3]
    num_sentences = max(len(sentences), 1)

    found = [phrase for phrase in _TRANSITION_WORDS if phrase in lower]
    ratio = len(found) / num_sentences
    score = _normalize(ratio, 0.0, 0.4)

    feedback = f"Transition phrases found: {len(found)} across {num_sentences} sentences. "
    if ratio < 0.1:
        feedback += "Very few linking words. Use transitions like 'however', 'therefore', 'furthermore'."
    elif ratio < 0.25:
        feedback += "Some transitions present. Adding more would improve logical flow."
    else:
        feedback += "Good use of transition phrases to guide the reader."

    return round(score, 1), feedback, found

def score_passive_voice(text):
    
    passive_pattern = re.compile(
         r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', re.IGNORECASE
    )
    sentences     = [s for s in _sentences(text) if len(s.strip()) > 3]
    num_sentences = max(len(sentences), 1)

    passive_matches = passive_pattern.findall(text)
    passive_count   = len(passive_matches)
    passive_ratio   = passive_count / num_sentences

    score = _normalize(1 - min(passive_ratio / 0.6, 1.0), 0.0, 1.0)
    feedback = (
        f"Passive constructions detected: {passive_count} "
        f"({passive_ratio * 100:.0f}% of sentences). "
    )
    if passive_ratio < 0.15:
        feedback += "Excellent active voice usage — writing feels direct and engaging."
    elif passive_ratio < 0.35:
        feedback += "Moderate passive voice. Try converting some sentences to active voice."
    else:
        feedback += "Heavy passive voice usage. Rewrite key sentences to use active voice."

    return round(score, 1), feedback

def score_argument(text):
    
    lower     = text.lower()
    words     = _words(text)
    num_words = max(len(words), 1)

    found = []
    for marker in _ARGUMENT_MARKERS:
        found.extend([marker] * lower.count(marker))

    density = (len(found) / num_words) * 100
    score   = _normalize(density, 0.0, 4.0)

    feedback = f"Evidence/argument markers: {len(found)} ({density:.1f} per 100 words). "
    if density < 0.5:
        feedback += "Essay lacks supporting evidence. Use 'because', 'for example', 'research shows'."
    elif density < 2.0:
        feedback += "Some evidence markers present. Consider adding more specific support."
    else:
        feedback += "Strong use of evidence and reasoning markers — well-argued essay."

    return round(score, 1), feedback

def compute_overall(readability, vocabulary, structure, spelling,
                    transitions, passive_voice, argument):
    weighted = (
        readability   * 0.16 +
        vocabulary    * 0.16 +
        structure     * 0.16 +
        spelling      * 0.16 +
        transitions   * 0.14 +
        passive_voice * 0.10 +
        argument      * 0.12
    )
    return round(max(1.0, min(10.0, weighted)), 1)

def annotate_text(text, misspelled_set=None, transition_words_found=None):
        
    annotations = []
    lower_text  = text.lower()

    all_words   = re.findall(r'\b[a-zA-Z]+\b', lower_text)
    word_freq   = Counter(w for w in all_words if w not in _STOP_WORDS and len(w) > 3)

    used_ranges = []

    def _overlaps(start, end):
         return any(s < end and e > start for s, e in used_ranges)

    def _claim(start, end):
        used_ranges.append((start, end))

    passive_re = re.compile(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', re.IGNORECASE)
    for m in passive_re.finditer(text):
        s, e = m.start(), m.end()
        if not _overlaps(s, e):
            annotations.append({"start": s, "end": e, "word": m.group(), "tag": "passive"})
            _claim(s, e)

    sorted_transitions = sorted(_TRANSITION_WORDS, key=len, reverse=True)
    for phrase in sorted_transitions:
        for m in re.finditer(re.escape(phrase), lower_text):
            s, e = m.start(), m.end()
            if not _overlaps(s, e):
                annotations.append({"start": s, "end": e, "word": text[s:e], "tag": "transition"})
                _claim(s, e)

    for m in re.finditer(r'\b[a-zA-Z]+\b', text):
        s, e       = m.start(), m.end()
        word       = m.group()
        word_lower = word.lower()

        if _overlaps(s, e):
            continue

        if misspelled_set and word_lower in misspelled_set:
            suggestion = spell.correction(word_lower)
            ann = {"start": s, "end": e, "word": word, "tag": "spelling"}
            if suggestion and suggestion != word_lower:
                ann["suggestion"] = suggestion
            annotations.append(ann)
            _claim(s, e)

        elif word_lower in _AWL_WORDS and len(word_lower) > 4:
            annotations.append({"start": s, "end": e, "word": word, "tag": "academic"})
            _claim(s, e)

        elif word_lower not in _STOP_WORDS and word_freq.get(word_lower, 0) > 3:
            annotations.append({"start": s, "end": e, "word": word, "tag": "overused"})
            _claim(s, e)

    annotations.sort(key=lambda a: a['start'])
    return annotations

def generate_suggestions(text, r_score, v_score, s_score, sp_score, ml_score,
                          tr_score, pv_score, arg_score,
                          transition_words_found, awl_ratio,
                          has_thesis, has_conclusion, misspelled_set):
    suggestions = []
    words     = _words(text)
    sentences = [s for s in _sentences(text) if len(s.strip()) > 3]

    if sp_score < 7.0 and misspelled_set:
        corrections = {}
        for w in list(misspelled_set)[:8]:
            c = spell.correction(w)
            if c and c != w:
                corrections[w] = c
        if corrections:
            examples = ", ".join([f'"{k}" → "{v}"' for k, v in list(corrections.items())[:5]])
            suggestions.append({
                "category": "Spelling", "priority": "high" if sp_score < 5 else "medium",
                "icon": "🔡", "title": "Fix spelling errors",
                "detail": f"Correct these misspelled words: {examples}."
            })

    if v_score < 6.5:
        if awl_ratio < 0.08:
            suggestions.append({
                "category": "Vocabulary", "priority": "medium",
                "icon": "🎓", "title": "Use more academic vocabulary",
                "detail": (
                    "Only {:.0f}% of your content words are from the Academic Word List. "
                    "Substitute common words with academic equivalents: e.g. 'show' → 'demonstrate', "
                    "'use' → 'utilise', 'important' → 'significant'.".format(awl_ratio * 100)
                )
            })
            total, unique = len(words), len(set(words))
        if total > 0 and unique / total < 0.45:
            stop = _STOP_WORDS
            freq = Counter(w for w in words if w not in stop)
            top_rep = [f'"{w}" ({c}×)' for w, c in freq.most_common(5)]
            suggestions.append({
                "category": "Vocabulary", "priority": "high" if v_score < 4 else "medium",
                "icon": "📚", "title": "Reduce word repetition",
                "detail": f"Overused words: {', '.join(top_rep)}. Use synonyms for variety."
            })
            if len(words) < 100:
            suggestions.append({
                "category": "Vocabulary", "priority": "medium",
                "icon": "✍️", "title": "Expand your essay",
                "detail": f"Only {len(words)} words — aim for 150–250+ for a well-developed argument."
            })

    if s_score < 6.5 or not has_thesis or not has_conclusion:
        avg_len = len(words) / max(len(sentences), 1)
        if avg_len < 8:
            suggestions.append({
                "category": "Structure", "priority": "medium",
                "icon": "📐", "title": "Develop your sentences",
                "detail": f"Average sentence is only {avg_len:.0f} words. Expand each idea with evidence, examples, or context."
            })
        elif avg_len > 30:
            suggestions.append({
                "category": "Structure", "priority": "medium",
                "icon": "✂️", "title": "Break up long sentences",
                "detail": f"Average sentence length is {avg_len:.0f} words — too dense. Split complex sentences for clarity."
            })
            if len(sentences) < 4:
            suggestions.append({
                "category": "Structure", "priority": "high",
                "icon": "🏗️", "title": "Add more paragraphs",
                "detail": "A strong essay needs an introduction, at least two body paragraphs, and a conclusion."
            })
        if not has_thesis:
            suggestions.append({
                "category": "Structure", "priority": "high",
                "icon": "📌", "title": "Add a clear thesis statement",
                "detail": "Your opening sentence should state your argument clearly. Try: 'This essay argues that…' or 'I contend that…'."
            })
        if not has_conclusion:
            suggestions.append({
                "category": "Structure", "priority": "medium",
                "icon": "🏁", "title": "Add a conclusion paragraph",
                "detail": "End with 'In conclusion…', 'To summarise…' or 'Overall…' followed by a restatement of your main argument."
            })

    if r_score < 6.0:
        try:
            grade = textstat.flesch_kincaid_grade(text)
            if grade > 14:
                suggestions.append({
                    "category": "Readability", "priority": "medium",
                    "icon": "🔭", "title": "Simplify complex phrasing",
                    "detail": f"Grade {grade:.0f} level — aim for Grade 10–12. Prefer active voice and concise sentences."
                })
            elif grade < 5:
                suggestions.append({
                    "category": "Readability", "priority": "low",
                    "icon": "🎓", "title": "Use more academic language",
                    "detail": "Introduce more sophisticated transitions and complex sentence structures."
                })
        except Exception:
            pass

    if tr_score < 6.5:
        missing = [w for w in ['however', 'therefore', 'furthermore', 'consequently', 'for example']
                    if w not in transition_words_found][:3]
        suggestions.append({
            "category": "Transitions", "priority": "medium" if tr_score < 4.0 else "low",
            "icon": "🔗", "title": "Add transition phrases",
            "detail": f"Connect ideas with linking words. Try adding: {', '.join(missing)}."
        })

    if pv_score < 5.5:
        suggestions.append({
            "category": "Style", "priority": "low",
            "icon": "💪", "title": "Reduce passive voice",
            "detail": "Convert passive to active: e.g. 'The essay was written by me' → 'I wrote the essay'."
        })

    if arg_score < 5.5:
        suggestions.append({
            "category": "Argument", "priority": "high" if arg_score < 3.5 else "medium",
            "icon": "🎯", "title": "Support your claims with evidence",
            "detail": "Use: 'because', 'for example', 'research shows', or 'this demonstrates' to back up arguments."
        })

    if ml_score < 6.0:
        suggestions.append({
            "category": "Coherence", "priority": "medium",
            "icon": "🧠", "title": "Strengthen logical flow",
            "detail": "Use: 'As a result', 'In contrast', 'Building on this', 'To conclude' to link ideas."
        })

    universal = [
        {"category": "General", "priority": "low", "icon": "💡",
          "title": "Read your essay aloud",
         "detail": "Reading aloud catches awkward phrasing and missing words."},
        {"category": "General", "priority": "low", "icon": "🔄",
         "title": "Revise and review",
         "detail": "Leave 30 minutes between writing and proofreading for fresh perspective."},
    ]
    suggestions.extend(universal[:2] if len(suggestions) < 3 else universal[:1])

    priority_order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: priority_order.get(s["priority"], 3))
    return suggestions  

def analyze_paragraphs(text):
    
    raw_paras = [p.strip() for p in text.strip().split('\n\n') if p.strip()]
    if not raw_paras:
        raw_paras = [text.strip()]

    results = []
    for i, para in enumerate(raw_paras[:10]):              
        words     = _words(para)
        sentences = [s for s in _sentences(para) if len(s.strip()) > 3]
        n_words   = len(words)
        n_sents   = max(len(sentences), 1)
        avg_len   = n_words / n_sents

        try:
            r_s, _ = score_readability(para) if n_words > 10 else (5.0, '')
        except Exception:
            r_s = 5.0           

        lower = para.lower()
        tr_found = [ph for ph in _TRANSITION_WORDS if ph in lower]
        tr_ratio = len(tr_found) / n_sents
        tr_s = min(10.0, 1 + tr_ratio / 0.4 * 9)

        passive_re = re.compile(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', re.IGNORECASE)
        pv_count = len(passive_re.findall(para))
        pv_ratio = pv_count / n_sents
        pv_s = max(1.0, 10.0 - pv_ratio * 15)

        content = [w for w in words if w not in _STOP_WORDS and len(w) > 3]
        awl_n = sum(1 for w in content if w in _AWL_WORDS)
        awl_r = awl_n / max(len(content), 1)

        para_score = round(r_s * 0.4 + tr_s * 0.3 + pv_s * 0.3, 1)
        para_score = max(1.0, min(10.0, para_score))

        if i == 0:
            role = "Introduction"
            elif i == len(raw_paras) - 1:
            role = "Conclusion"
        else:
            role = f"Body ¶{i}"

        results.append({
            "index":      i + 1,
            "role":       role,
            "preview":    para[:70].replace('\n', ' ') + ('…' if len(para) > 70 else ''),
            "words":      n_words,
            "sentences":  len(sentences),
            "avg_len":    round(avg_len, 1),
            "transitions":len(tr_found),
            "awl_pct":    round(awl_r * 100, 1),
            "score":      para_score,
        })
    return results

def analyze_essay(text, ml_score=5.0):

    r_score,  r_feedback                       = score_readability(text)
    v_score,  v_feedback, awl_count, awl_ratio = score_vocabulary(text)
    s_score,  s_feedback, has_thesis, has_conclusion = score_structure(text)
    sp_score, sp_feedback, misspelled_set      = score_spelling(text)
    tr_score, tr_feedback, transition_found    = score_transitions(text)
    pv_score, pv_feedback                      = score_passive_voice(text)
    arg_score, arg_feedback                    = score_argument(text)

    ml_clamped = round(max(1.0, min(10.0, float(ml_score))), 1)

    overall = compute_overall(
        r_score, v_score, s_score, sp_score,
        tr_score, pv_score, arg_score
    )

    dims = {
        "readability":  {"score": r_score,    "label": "Readability",  "feedback": r_feedback,   "icon": "📖"},
        "vocabulary":   {"score": v_score,    "label": "Vocabulary",   "feedback": v_feedback,   "icon": "🔤",
                         "detail": {"awl_count": awl_count, "awl_pct": round(awl_ratio * 100, 1)}},
       "structure":    {"score": s_score,    "label": "Structure",    "feedback": s_feedback,   "icon": "📐",
                         "detail": {"has_thesis": has_thesis, "has_conclusion": has_conclusion}},
        "spelling":     {"score": sp_score,   "label": "Spelling",     "feedback": sp_feedback,  "icon": "✅",
                         "detail": {"errors": list(misspelled_set)[:10]}},
        "transitions":  {"score": tr_score,   "label": "Transitions",  "feedback": tr_feedback,  "icon": "🔗",
                         "detail": {"found": list(transition_found)[:10]}},
        "passive_voice":{"score": pv_score,   "label": "Active Voice", "feedback": pv_feedback,  "icon": "💪"},
        "argument":     {"score": arg_score,  "label": "Argument",     "feedback": arg_feedback, "icon": "🎯"},
        "coherence":    {"score": ml_clamped, "label": "Coherence",
                         "feedback": "Assessed by neural network analysis of writing style.",
                         "icon": "🧠"},
    }

    annotations = annotate_text(
        text,
        misspelled_set=misspelled_set,
        transition_words_found=transition_found
    )

    suggestions = generate_suggestions(
        text,
        r_score, v_score, s_score, sp_score, ml_clamped,
        tr_score, pv_score, arg_score,
        transition_found, awl_ratio,
        has_thesis, has_conclusion, misspelled_set
    )

    words     = _words(text)
    sentences = [s for s in _sentences(text) if len(s.strip()) > 3]
    try:
        grade = textstat.flesch_kincaid_grade(text)
    except Exception:
        grade = 0.0
    reading_time = max(1, round(len(words) / 238))                     

    stats = {
        "word_count":       len(words),
        "sentence_count":   len(sentences),
        "paragraph_count":  len([p for p in text.strip().split('\n\n') if p.strip()]),
        "reading_time_min": reading_time,
        "grade_level":      round(grade, 1),
        "awl_pct":          round(awl_ratio * 100, 1),
        "spelling_errors":  len(misspelled_set),
    }

    paragraphs = analyze_paragraphs(text)

    return {
        "overall_score": overall,
        "dimensions":    dims,
        "suggestions":   suggestions,
        "annotations":   annotations,
        "stats":         stats,
        "paragraphs":    paragraphs,
    }

ssl._create_default_https_context = ssl._create_unverified_context

reader = easyocr.Reader(['en'], gpu=False)

def _preprocess(img):
        
    h, w = img.shape[:2]

    min_dim = 1200
    if max(h, w) < min_dim:
        scale = min_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def extract_text_from_buffer(file_bytes):
    
    try:

        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode the provided image bytes.")

        preprocessed = _preprocess(img)

        print("Running EasyOCR with enhanced preprocessing...")

        results = reader.readtext(
            preprocessed,
            detail=1,
            paragraph=False,                                                   
            text_threshold=0.5,                                                  
            low_text=0.35,                                                          
            link_threshold=0.35,                                            
            width_ths=0.8,                                                
            height_ths=0.5,    
            min_size=10,                                                  
            slope_ths=0.2,                                             
        )

        if not results:
            return ""

        def sort_key(r):
            bbox = r[0]

            top_y = min(pt[1] for pt in bbox)
            left_x = min(pt[0] for pt in bbox)

            row_band = int(top_y / 30)
            return (row_band, left_x)

        results_sorted = sorted(results, key=sort_key)

        MIN_CONFIDENCE = 0.3
        lines_data = [(bbox, text, conf) for (bbox, text, conf) in results_sorted if conf >= MIN_CONFIDENCE]

        if not lines_data:
            return ""

        extracted_text = ""
        prev_y = None
        
        for bbox, text, conf in lines_data:
            top_y = min(pt[1] for pt in bbox)

            if prev_y is None:
                extracted_text += text
            else:
                y_diff = top_y - prev_y

                if y_diff > 40:
                    extracted_text += "\n\n" + text

                elif y_diff > 20:
                    extracted_text += "\n" + text

                else:
                    extracted_text += " " + text
            
            prev_y = top_y

        extracted_text = extracted_text.strip()
        print(f"Extracted {len(lines_data)} text regions, {len(extracted_text)} characters.")

        return extracted_text