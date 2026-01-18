# Autonomous Vehicle TRL Assessment System

## Overview

The **AV TRL Assessment System** evaluates Autonomous Vehicle readiness across **5 critical dimensions**: Infrastructure, Technology, Data & Connectivity, Regulatory, and Societal readiness.

## Installation

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
```

### Step 1: Install Dependencies
```bash
# Core dependencies
pip install flask==2.3.0 flask-cors==4.0.0 experta==1.9.4

# File processing
pip install PyPDF2==3.0.0 python-docx==0.8.11 werkzeug==2.3.0

# AI integration
pip install google-generativeai==0.3.0


### Step 2: Configure API Keys

Add your key to the main file (hybird_expert_system.py):
```env
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your_flask_secret_key_here
```

**Get Gemini API Key:** [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## Quick Start

### 1. Launch the Application
```bash
python hybird_expert_system.py
```
