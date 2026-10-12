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