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