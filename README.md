# MindBalance 🧠

**An AI-driven mental health and wellbeing web application** built as a Final Year Project at the University of Roehampton.

MindBalance helps users track their emotional state over time, applying machine learning to surface personalised wellbeing and academic recommendations. The system processes longitudinal mood data through an end-to-end Python pipeline — from user input, through ML transformation, to visual insight.

---

## Features

- 🔍 **Emotion Analysis** — classifies user-reported mood and emotional state using a trained ML model
- 📈 **Mood Tracking Over Time** — captures and stores longitudinal emotional data, enabling trend analysis
- 💡 **Personalised Recommendations** — generates tailored wellbeing and academic suggestions based on detected patterns
- 📊 **Data Visualisation** — surfaces emotional trend graphs to help users understand their mental health patterns
- ✅ **Automated Data Validation** — flags inconsistent or incomplete entries before they enter the pipeline
- 🔒 **User Data Storage** — SQLite backend stores interaction history securely with a Flask API layer

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask, SQLite, SQL |
| Machine Learning | Scikit-learn, ML classification models |
| Data Pipeline | Python (ingestion → transformation → output) |
| Visualisation | Matplotlib / data visualisation libraries |
| Frontend | HTML, CSS |
| Version Control | Git / GitHub |

---

## Project Structure

```
MindBalance/
├── Backend/          # Flask app, API routes, ML model, data pipeline
├── Frontend/         # HTML/CSS templates and UI components  
├── Graphs/           # Generated emotional trend visualisations
├── Final Year Project Presentation ...
└── BSc Project Report Template_v...
```

---

## How It Works

1. **User Input** — the user logs their current mood and emotional state via the web interface
2. **Data Ingestion** — the Flask backend receives and validates the input, storing it in SQLite
3. **ML Processing** — a trained classification model analyses the emotional data and identifies patterns
4. **Insight Generation** — personalised recommendations are produced based on the user's recent trend
5. **Visualisation** — emotional trends are plotted and returned to the user dashboard

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/AlizaRaza/MindBalance.git
cd MindBalance

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
cd Backend
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## Background

MindBalance was developed as a Final Year Project exploring the intersection of machine learning and mental health support. The project demonstrates how AI can be applied sensitively and interpretably in health-facing contexts — tracking emotional signals over time and translating them into actionable, human-readable insights.

This work informed subsequent research into Explainable AI (XAI) for clinical health applications, including an ongoing MSc project applying Grad-CAM visualisation to ASD detection via facial image analysis.

---

## Author

**Aliza Raza** — MSc AI & Robotics, Queen Mary University of London  
[LinkedIn](https://linkedin.com/in/aliza-raza-b35a391b6) · [GitHub](https://github.com/AlizaRaza)
