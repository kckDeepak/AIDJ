# AI DJ Mixing System

## Overview

The **AI DJ Mixing System** is a professional, AI-powered DJ mixing pipeline that creates seamless, radio-quality mixes from your local music library. It features a modern **Next.js Frontend**, a **FastAPI Backend**, and a sophisticated **Python Audio Pipeline**.

Using advanced audio analysis, machine learning, and OpenAI's GPT models, it automatically selects tracks, detects optimal transition points, and creates professional DJ transitions with perfect beat-grid alignment and energy flow management.

---

## 🏗️ Architecture

The system consists of three main components:

1.  **Frontend (Next.js 16)**:
    *   Modern web interface for users to input prompts (e.g., "Play some upbeat 80s disco").
    *   Visualizes mix progress and song structure.
    *   Built with React 19, Tailwind CSS, and Three.js/Drei for 3D elements.

2.  **Backend (FastAPI)**:
    *   REST API to handle song uploads, mix requests, and management.
    *   **WebSocket** server for real-time progress updates to the frontend.
    *   Orchestrates the heavy python pipeline tasks in the background.

3.  **Core Pipeline (Python)**:
    *   The "brain" of the DJ. Handles specific stages of processing:
    *   **Analysis**: Librosa & OpenAI for BPM, Key, and Structure detection.
    *   **Planning**: Generates a cue sheet (mixing plan) based on harmonic mixing rules.
    *   **Mixing**: DSP processing (EQ, Time-stretch, Crossfade) to create the final MP3.

---

## 📁 Project Structure

Here is a detailed breakdown of the files and folders in this repository:

```
AI-DJ-Mixing-System/
│
├── frontend/                       # ⚛️ Next.js Web App
│   ├── src/                        # Component source code
│   ├── package.json                # Frontend dependencies
│   └── ...
│
├── backend/                        # 🚀 FastAPI Backend Server
│   ├── routers/                    # API Endpoints
│   │   ├── mix.py                  # Generation endpoints & WebSockets
│   │   ├── songs.py                # Song management
│   │   └── upload.py               # File upload handling
│   ├── services/                   # Business Logic
│   │   └── pipeline_runner.py      # Background task runner
│   ├── main.py                     # Server entry point
│   └── requirements.txt            # Backend-specific python libs
│
├── songs/                          # 🎵 Input: Place your MP3 files here
├── output/                         # 📤 Output: Where final mixes are saved
├── notes/                          # 💾 Cache: Stores analysis data to speed up re-runs
│
├── requirements.txt                # 📦 Global Python dependencies
├── run_pipeline.py                 # ▶️ CLI Entry Point (Run pipeline without UI)
│
├── track_analysis_openai_approach.py  # Stage 1: Intelligent Song Selection
├── bpm_lookup.py                      # Stage 2: BPM & Metadata Extraction
├── structure_detector.py              # Stage 3: Chorus & Transition Detection
├── generate_mixing_plan.py            # Stage 4: DJ Set Planning
├── mixing_engine.py                   # Stage 5: Audio Processing & Mixing
│
└── .env                            # 🔑 Environment Variables (API Keys)
```

---

## 🚀 Setup & Installation

### Prerequisites

*   **Python 3.8+**
*   **Node.js 18+** (for Frontend)
*   **FFmpeg** (Required for audio processing)
    *   Windows: [Download](https://ffmpeg.org/download.html), extract, and add `bin` folder to your System PATH.
    *   Mac: `brew install ffmpeg`
    *   Linux: `sudo apt install ffmpeg`

### 1. Environment Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-your-openai-key-here
```

### 2. Install Python Dependencies

```bash
# It is recommended to use a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
```

---

## 🏃‍♂️ How to Run

You can run the system in two ways: as a **Full Web Application** (Recommended) or as a **Standalone CLI Tool**.

### Option A: Full Web Application

This runs the React UI and the API Server.

**1. Start the Backend Server**
Open a terminal in the root folder:
```bash
# Make sure your virtual env is active
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*Server will start at `http://localhost:8000`*

**2. Start the Frontend**
Open a **new** terminal, navigate to `frontend/`:
```bash
cd frontend
npm run dev
```
*Frontend will run at `http://localhost:3000`*

**3. Use the App**
*   Open `http://localhost:3000` in your browser.
*   Upload songs or ensure MP3s are in the `songs/` folder.
*   Type a prompt (e.g., "Mix 5 upbeat songs") and watch it work!

---

### Option B: Standalone CLI (Command Line)

If you just want to generate a mix without the UI:

1.  Place your MP3s in the `songs/` folder.
2.  Run the pipeline script:

```bash
python run_pipeline.py
```

3.  Follow the prompts to enter your request.
4.  The final mix will be generated in `output/mix.mp3`.

---

## 🧠 Core Pipeline Explained

When you request a mix, the Python pipeline executes these 5 stages:

1.  **Selection (`track_analysis_openai_approach.py`)**:
    *   Uses OpenAI GPT-4o to scan your `songs/` folder and select tracks that match your text prompt (mood, genre, specific artist).

2.  **Analysis (`bpm_lookup.py`)**:
    *   Loads audio using `librosa`.
    *   Calculates exact BPM and Energy levels.
    *   Detects Musical Key (Camelot Wheel) for harmonic mixing.

3.  **Structure (`structure_detector.py`)**:
    *   Uses OpenAI Whisper (or energy analysis) to find vocals.
    *   Identifies "safe" transition points (verse/chorus boundaries) to avoid vocal clashes.

4.  **Planning (`generate_mixing_plan.py`)**:
    *   The "DJ Brain". It decides exactly when to mix out Song A and mix in Song B.
    *   Matches beats, aligns phrases (8-bar blocks), and checks key compatibility.

5.  **Mixing (`mixing_engine.py`)**:
    *   The "Hands". Executes the plan using Digital Signal Processing.
    *   Applies Low-pass/High-pass EQ filters for smooth blends.
    *   Time-stretches audio to match tempo without changing pitch.

---

## 🛠️ Troubleshooting

**"FFmpeg not found"** or **"FileNotFoundError: [WinError 2]"**
*   FFmpeg is missing from your system PATH. Download it and restart your terminal.

**"OpenAI API key not found"**
*   Make sure you have a `.env` file in the root with `OPENAI_API_KEY=...`.

**"Module not found"**
*   Run `pip install -r requirements.txt` again.
