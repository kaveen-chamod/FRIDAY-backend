@'
# ⚙️ F.R.I.D.A.Y. AI

### The Cloud-Native, Autonomous Hybrid AI Assistant & Cyber-Physical Operating Layer
**Engineered by Kaveen Chamod**  
*(Based on the core architectural principles of Project A.E.G.I.S. & F.R.I.D.A.Y. — Inspired by Marvel's Iron Man)*



# 🚀 F.R.I.D.A.Y. AI – Backend Repository Setup & Deployment

The F.R.I.D.A.Y. Backend is built using Python and FastAPI. It provides the core AI processing, authentication, security, memory management, robotics integration, and real-time communication services.

## 1. Navigate to the Backend Directory

Open PowerShell and navigate to the main `JarvisAI` backend directory.

```powershell
cd "D:\MERN_development\Jarvis-AI-main\Jarvis-AI-main\JarvisAI"
```

**Important:** Make sure you are in the Backend root directory, not inside the `Front End` folder.

## 2. Initialize Git

```powershell
git init
git branch -M main
```

## 3. Connect the GitHub Backend Repository

Create a separate GitHub repository for the backend.

Example repository name: `FRIDAY-backend`

Then connect it:

```powershell
git remote add origin YOUR_BACKEND_REPOSITORY_URL
```

Replace `YOUR_BACKEND_REPOSITORY_URL` with your actual GitHub repository URL.

If the remote is already configured, use:

```powershell
git remote set-url origin YOUR_BACKEND_REPOSITORY_URL
```

## 4. Add Backend Files

```powershell
git add .
```

## 5. Commit Backend Files

```powershell
git commit -m "Initial commit - F.R.I.D.A.Y. AI Backend"
```

## 6. Push Backend to GitHub

```powershell
git push -u origin main
```

## 7. Install Python Dependencies

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install the required dependencies:

```powershell
pip install -r requirements-cloud.txt
```

## 8. Configure Environment Variables

Configure the required environment variables before starting the backend.

```powershell
$env:FRIDAY_GEMINI_API_KEY = "your-gemini-api-key"
$env:FRIDAY_API_TOKEN = "your-secure-api-token"
$env:FRIDAY_TELEGRAM_BOT_TOKEN = "your-telegram-bot-token"
$env:FRIDAY_TELEGRAM_ALLOWED_USER_IDS = "your-telegram-user-id"
$env:FRIDAY_TELEGRAM_WEBHOOK_URL = "your-telegram-webhook-url"
$env:FRIDAY_TELEGRAM_WEBHOOK_SECRET = "your-webhook-secret"
```

**Security Note:** Never upload API keys, tokens, passwords, or `.env` files to GitHub.

## 9. Start the Backend Server

```powershell
python -m app.main
```

## 10. Run Automated Tests

To verify the backend functionality:

```powershell
python -m unittest discover -s tests -v
```

## 🔄 Updating the Backend Repository

Whenever you make changes to the backend source code, use:

```powershell
git add .
git commit -m "Update F.R.I.D.A.Y. Backend"
git push origin main
```

## ⚠️ Important Notes

- Keep the Backend and Frontend in separate GitHub repositories.
- Do not upload the `.venv/` directory.
- Do not upload `__pycache__/` directories.
- Never expose API keys, access tokens, or private credentials.
- Maintain an updated `requirements-cloud.txt` file.
- Make sure the backend configuration is ready before cloud deployment.

---

**F.R.I.D.A.Y. AI – Backend**

Developed by Kaveen Chamod.

*Tactical Intelligence. Absolute Loyalty. Zero Compromise.*
==========================================================
==========================================================

> **Hear. Speak. Think. Protect. Simulate. Act.**

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_Headless-009688.svg)](https://fastapi.tiangolo.com/)
[![React PWA](https://img.shields.io/badge/Frontend-React_18_Vite_PWA-61DAFB.svg)](https://react.dev/)
[![Gemini Live](https://img.shields.io/badge/Voice_Engine-Gemini_Live_Aoede-8E75C2.svg)](https://ai.google.dev/)
[![Test Suite](https://img.shields.io/badge/Tests-274%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-orange.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

# ✨ What is F.R.I.D.A.Y.?

**F.R.I.D.A.Y.** (Female Replacement Intelligent Digital Assistant Youth) යනු සාමාන්‍ය Chatbot කෙනෙකුගෙන් ඔබ්බට ගිය, සැබෑ ලෝකයේ පරිසරය, පවුලේ සාමාජිකයන්, ස්මාර්ට් නිවස, රොබෝ තාක්ෂණය සහ සයිබර් අවකාශය ස්වයංක්‍රීයව කළමනාකරණය කරන **අධි-තාක්ෂණික පෞද්ගලික බුද්ධිමය මෙහෙයුම් පද්ධතියකි (Cyber-Physical AI Companion)**.

මෙම පද්ධතිය පරිගණක තිරයකට හෝ එක් මෙහෙයුම් පද්ධතියකට සීමා නොවී, **Cloud-First Headless Architecture** එකක් මත පැය 24 පුරාම ධාවනය වේ. නිවසේ කිසිදු පරිගණකයක් On කර නොමැති වුවද, කුඩා **ESP32 Ambient Satellite** එකක්, **Telegram Bot** එකක් සහ **Tactical Web HUD (PWA)** එකක් හරහා ලෝකයේ කොහේ සිටියත් ක්‍රියාත්මක වේ.

---

# 🚀 Core Capabilities at a Glance

### 🎙️ 1. Sub-Second Real-Time Voice & Bilingual Fluency
* **Gemini Live Native Audio (`Aoede` Voice):** කෘත්‍රිම රොබෝ හඬවල් වෙනුවට, මිනිස් හැඟීම්, හුස්ම ගැනීම් සහ නිවැරදි විරාම සහිතව සජීවීව කතා කරන Google Gemini හි සුප්‍රසිද්ධ `Aoede` හඬ භාවිතය.
* **Bilingual Spoken Speech:** පොත් බසින් තොරව, ස්වභාවික කතා කරන සිංහල, Singlish සහ English භාෂා ත්‍රිත්වයෙන්ම තත්පර 0.5ක අතිශය අඩු ප්‍රමාදයකින් (Sub-second latency) ප්‍රතිචාර දැක්වීම.
* **Resilient Multi-Tier Fallback:** Google Voice Quota සීමා වුවද බිඳ නොවැටී `Edge-TTS` (Emily/Thilini) සහ Text Fallback වෙත ස්වයංක්‍රීයව මාරු වීම (Zero-Crash Guarantee).

### 👥 2. Multi-Speaker Family Biometrics & RBAC Hierarchy
කටහඬේ සංඛ්‍යාත (Speaker Embeddings, Pitch $F_0$, Formants) මගින් පවුලේ අය වෙන් වෙන් වශයෙන් හඳුනාගෙන සුදුසු බලතල ලබා දීම:
* **MASTER / Sir:** පද්ධතියේ 100% ක් පූර්ණ පාලනය (Root authority).
* **FATHER / Sir & MOTHER / Madam:** ගෘහස්ථ පාලනය (Smart Home), කාලසටහන් සහ අනුප්‍රාප්තික බලය අනුමත කිරීමේ (Parental Guardian) බලතල.
* **BROTHER & SISTER:** සාමාන්‍ය කතාබහ, අධ්‍යාපනික Quizzes සහ අනුමත මාධ්‍ය පාලනය.
* **GUEST:** සාමාන්‍ය දැනුම සහ කාලගුණය පමණි. පෞද්ගලික ලිපිගොනු සහ Home Automation සම්පූර්ණයෙන්ම අවහිරයි.

### 🛡️ 3. Master Passphrase Security Gate & Safe Succession Protocol
* **Root Protection Gate:** AI එකට තමන්ගේම ආරක්ෂක කේත වෙනස් කිරීමට කිසිසේත්ම ඉඩ නොදෙන අතර, සියලු සංවේදී තීරණ වලට **Master Passphrase (PBKDF2-HMAC-SHA256)** එක අනිවාර්ය වේ.
* **Dead Man's Switch (හදිසි අනුප්‍රාප්තික දාමය):**
  $$\text{Master} \longrightarrow \text{Brother (දෙමාපිය අනුමැතියෙන්)} \longrightarrow \text{Sister} \longrightarrow \text{Sanctuary Lockdown}$$
  Master දිගු කලක් අක්‍රිය වුවහොත්, දෙමාපියන්ගේ Voice අනුමැතිය සහ රහස් Passphrase මගින් බලය මල්ලිට හෝ නංගිට මාරු වේ. Master නැවත පැමිණි සැනින් පාලනය යළි ලබා ගත හැක (Master Reclaim).

### 🔮 4. Digital Twin & Predictive Simulation Engine
* **Virtual Device Shadow:** RC Car එක, Smart Switches සහ පරිපථ වල සජීවී තත්ත්වය පරිගණකය තුළ අතථ්‍යව (Virtual Twin) පවත්වා ගනී.
* **Pre-Actuation Simulation:** සැබෑ ලෝකයේ විධානයක් ක්‍රියාත්මක කිරීමට පෙර, Virtual Twin එක තුළ Monte Carlo Simulations ($N \ge 1,000$) දුවවා අනතුරු හෝ Short-circuit අවදානම් කලින්ම ගණනය කර අවදානම් විධාන වළක්වයි.

### 👁️ 5. Proactive Ambient Observer & Self-Reflection
* **Unprompted Intervention:** ඔබ අතින් පරිපථයක් වැරදියට එකලස් වෙද්දී හෝ, පවුලේ සාමාජිකයන් අතර දිනය/කාලසටහනක වැරදීමක් සිදුවූ විට, කිසිවෙකුත් AI එකෙන් නොඇසුවද ආචාරශීලීව පසුබිමෙන් මැදිහත් වී නිවැරදි කරයි.
* **Autonomous Self-Reflection:** තමන් දුන් අනාවැකි සැබෑ ප්‍රතිඵල සමග සසඳා බලා (Brier calibration), තමන්ගේ වැරදි වලින් ස්වයංක්‍රීයව ඉගෙන ගනී.

### 🌐 6. Advanced Cyber Defense Subsystem
* **Anti-SSRF Egress Connector:** පද්ධතියෙන් පිටතට යන සියලුම Web Requests වල DNS validate කර, Private/Reserved IPs සහ Cloud Metadata endpoints (`169.254.169.254`) අවහිර කරයි.
* **Threat Matrix & SAFE_AUTO Orchestrator:** අනවසර brute-force හෝ IP ප්‍රහාර වලදී ස්වයංක්‍රීයව Temporary Rate-Limiting සහ IP Blocking ක්‍රියාත්මක කර, Telegram හරහා Master ට Alerts එවයි.

### 🏎️ 7. Modular Robotics & Spatial Navigation (RC Car & Drone HAL)
* **Dedicated Package (`app/robotics/`):** $A^*$ Pathfinding මගින් සාලයේ බාධක මගහැර Waypoints ඔස්සේ ගමන් කිරීම.
* **Tri-Mode Safety:** සෙන්ටිමීටර 15 ට වඩා බාධකයක් ළං වූ විට Hardware මට්ටමින් Emergency Braking සිදුවේ.
* **Drone-Ready HAL:** අනාගතයේදී Mini Drones (Tello/ESP32) එක් කිරීමට `BaseVehicle` abstraction එකෙන් සූදානම් කර ඇත.

### 🏠 8. Ambient Presence & Auto-Sleep Engine (Zero-Desktop Hardware)
* නිවසේ කිසිදු PC එකක් On කර තැබීමට අවශ්‍ය නැත. සාලයේ ඇති **ESP32-S3 Satellite** එකක් සහ Wi-Fi Router එකක් පමණක් ප්‍රමාණවත්ය.
* **PIR & Camera Fusion:** කාමරයට කෙනෙක් ආ විට පද්ධතිය අවදි වී, කාමරයේ කෙනෙක් සිටින තාක් කල් කතා කරන දේට සවන් දෙයි. කිසිවෙකුත් නැති විට **විනාඩි 15කින් Mic එක Hardware මට්ටමින් Mute වී Auto-Sleep වේ**.

### 📱 9. Decoupled Tactical Web HUD (React PWA)
* වෙනම `Front End/` ෆෝල්ඩරයක Vercel වෙත Deploy කළ හැකි standalone React/Vite PWA Dashboard එක.
* **Custom Tactical Color Palette:**
  - `#141414` (Sooty) — Deep Canvas
  - `#282828` (Dire Wolf) — Panels
  - `#3f3835` (Brownish Black) — Elevated Cards
  - `#4e4e4e` (Black Oak) — Grid & Dividers
  - `#a1887d` (Velvety Chestnut) — Secondary Badges
  - `#b55933` (Fiery Orange) — High-Intensity Arc Reactor Glow & Actions

---

# 🏗️ Architecture & Component Layout

```text
       ┌─────────────────────────────────────────────────────────────┐
       │                     CLIENT INTERFACES                       │
       │  - Tactical Web HUD (React / Vite PWA - Vercel Deployed)    │
       │  - Telegram Bot (Webhook Mode + static-ffmpeg)              │
       │  - Ambient ESP32-S3 Voice Satellite (Microphone + Speaker)  │
       └──────────────────────────────┬──────────────────────────────┘
                                      │ (HTTPS / WSS TLS 1.3)
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    F.R.I.D.A.Y. CLOUD BACKEND (FastAPI)                    │
│                                                                            │
│  ├── 🧠 Cognitive Core: Gemini 3.5 Flash Lite + Gemini Live (Aoede)        │
│  ├── 🛡️ Security Core: Master Passphrase Gate + RBAC + Succession Engine   │
│  ├── 🌐 Cyber Defense: Anti-SSRF + Threat Matrix + SAFE_AUTO Orchestrator  │
│  ├── 🔮 Simulation Engine: Digital Twin + Monte Carlo Risk Simulator       │
│  ├── 🏎️ Robotics HAL: A* Spatial Mapping + RC Car / Drone Drivers          │
│  ├── 👁️ Ambient Engine: Presence Lifecycle (15m Sleep Timer)               │
│  └── 🧰 Native Tool Registry: 25 Actions + 8 Plugins (Weather, Files, etc) │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │ (Cloud MQTT - HiveMQ / Wi-Fi)
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                 PHYSICAL HARDWARE & IOT                     │
       │  - Smart Robot Car: ESP32 + L298N Driver + HC-SR04 Sensor   │
       │  - Smart Home Relays: Tuya / Sonoff Wi-Fi Switches          │
       │  - Ambient Node: ESP32-S3 + INMP441 Mic + MAX98357A DAC     │
       └─────────────────────────────────────────────────────────────┘


       JarvisAI/
├── app/
│   ├── main.py                  # Headless FastAPI Entrypoint (REST & WebSockets)
│   ├── agent.py                 # Master Cognitive Router (Sinhala/English Persona)
│   ├── access_control.py        # RBAC, Family Roles, Signed Speaker Tokens
│   ├── voice_profiles.py        # Multi-Sample Voice Biometrics & Encryption
│   ├── succession.py            # Hierarchical Emergency Succession Engine
│   ├── digital_twin.py          # Device Shadows & Virtual Simulators
│   ├── predictive_engine.py     # Monte Carlo Risk & Probability Scoring
│   ├── ambient_observer.py      # Proactive Interception & Context Evaluation
│   ├── self_reflection.py       # Calibration & Autonomous Learning Ledger
│   ├── web_bridge.py            # Headless Player Bridge for Native Tools
│   ├── telegram_bot.py          # Webhook Telegram Service
│   │
│   ├── defense/                 # Advanced Cyber Defense Package
│   │   ├── asset_registry.py    # Zero Trust Authorized Asset Management
│   │   ├── secure_access.py     # Anti-SSRF & Low-Exposure Egress Client
│   │   ├── threat_matrix.py     # 6-Category Quantitative Threat Scorer
│   │   ├── orchestrator.py      # SAFE_AUTO Mitigation & Rollback Daemon
│   │   └── forensics.py         # Append-only SHA-256 Incident Ledger
│   │
│   ├── robotics/                # Robotics & Spatial Navigation Package
│   │   ├── base_vehicle.py      # Hardware Abstraction Layer (HAL)
│   │   ├── spatial_map.py       # 2D Grid Waypoints & A* Pathfinding
│   │   ├── collision_guard.py   # Obstacle Threshold (<15cm) Braking Gate
│   │   ├── tracking_engine.py   # Headless ArUco Marker Optical Tracker
│   │   ├── rc_car.py            # Concrete ESP32 RC Car Driver
│   │   └── drone.py             # Extensible Drone Flight Driver
│   │
│   └── presence/                # Ambient Presence Subsystem
│       └── presence_engine.py   # PIR & Vision Presence State Machine (15m Sleep)
│
├── actions/                     # 25+ Built-in Tools (Weather, YouTube, Files, System...)
├── plugins/                     # 8 Extensible Plugins (Gmail, Calendar, Tuya, Printer...)
├── core/                        # Protected Core (Security, Sandbox, Confirm, PathGuard)
├── memory/                      # SQLite Long-Term Memory, Scored Retrieval & Ledgers
│
├── firmware/                    # Hardware Microcontroller Firmware
│   └── esp32_ambient_satellite/ # ESP32-S3 I2S Mic + Speaker PlatformIO C++ Project
│
├── Front End/                   # Standalone React + Vite PWA Tactical HUD (Sooty/Orange)
│   ├── src/                     # HoloCore, DigitalTwinMap, CyberRadar, Telemetry...
│   └── package.json             # Independent Vercel/Netlify Deployment Manifest
│
├── tests/                       # 274 Automated Unit & Regression Tests (100% Pass)
└── requirements-cloud.txt       # Native Cloud Dependencies (with static-ffmpeg)

ESP32-S3 Hardware Pinout (Ambient Satellite)
නිවසේදී කිසිදු පරිගණකයක් නොමැතිව F.R.I.D.A.Y. සමඟ කාමරයේ සිට කතා කිරීමට පහත පරිපථ සටහන භාවිතා කරන්න:
සංරචකය (Component)	Module Pin	ESP32-S3 Pin	කාර්යභාරය
PIR Motion Sensor	OUT	GPIO 13	චලනය දුටු සැනින් System එක Wake-up කිරීම
INMP441 Microphone	SCK (BCLK)	GPIO 14	I2S Bit Clock
WS (LRCK)	GPIO 15	Word Select (Left/Right)
SD (DOUT)	GPIO 32	Audio Stream to Cloud
VDD / GND	3.3V / GND	Power Supply
MAX98357A Amplifier	BCLK	GPIO 26	Audio Bit Clock
LRC	GPIO 16	Word Select
DIN	GPIO 17	Audio Data Input from Cloud
VIN / GND	5V / GND	Speaker Power
⚡ Quick Start & Deployment Guide
1. Cloud Backend Setup (Render / Railway / Koyeb)
Docker අවශ්‍ය නැත. Python 3.12 පරිසරයක කෙලින්ම ධාවනය කළ හැක:


# 1. Virtual Environment සාදා සක්‍රිය කරගන්න
python -m venv .venv
.\.venv\Scripts\activate

# 2. Cloud Dependencies ස්ථාපනය කරන්න (static-ffmpeg ඇතුළුව)
pip install -r requirements-cloud.txt

# 3. Environment Variables සකසන්න
$env:FRIDAY_GEMINI_API_KEY = "your-gemini-api-key"
$env:FRIDAY_API_TOKEN = (python -c "import secrets; print(secrets.token_urlsafe(48))")
$env:FRIDAY_TELEGRAM_BOT_TOKEN = "your-telegram-bot-token"
$env:FRIDAY_TELEGRAM_ALLOWED_USER_IDS = "123456789"
$env:FRIDAY_TELEGRAM_WEBHOOK_URL = "https://your-app.onrender.com/api/telegram-webhook"
$env:FRIDAY_TELEGRAM_WEBHOOK_SECRET = (python -c "import secrets; print(secrets.token_urlsafe(32))")

# 4. Backend Server එක Start කරන්න
python -m app.main




cd "Front End"
npm install
npm run dev


Production Build සඳහා: npm run build 
Vercel වෙත Deploy කිරීමට: Front End/ ෆෝල්ඩරය වෙනම GitHub Repo එකකට දමා Vercel වෙතින් Deploy කරන්න.



3. Master Passphrase එක පළමුවරට සැකසීම
Server එක Start වූ පසු, පරිපාලක රහස් මුරපදය සැකසීමට Request එකක් යවන්න:



$headers = @{
    "Authorization" = "Bearer $env:FRIDAY_API_TOKEN"
    "Content-Type" = "application/json"
}
$body = @{ "passphrase" = "YourSuperSecretMasterPassphrase123!" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/master/setup" -Method Post -Headers $headers -Body $body



Verification & Test Suite
පද්ධතියේ ආරක්ෂාව, Succession, Webhook, සහ Robotics ඇතුළු සියලුම කොටස් ආවරණය වන පරිදි Tests 274ක් පද්ධතිය තුළ ලියා පරීක්ෂා කර ඇත:
code
Powershell
python -m unittest discover -s tests -v
ප්‍රතිඵලය: Ran 274 tests in 42.15s — OK (100% Passed, 0 Failures).


 Safety, Privacy & Ethics
Local-First Privacy: නිවසේ Audio 24 පැය පුරා Record වීමක් සිදු නොවේ. කාමරයේ කිසිවෙකුත් නැති විට මිනිත්තු 15කින් Mic එක Hardware මට්ටමින් Mute වේ.
Strict Anti-Exploit Policy: ආරක්ෂක මෙවලම් ක්‍රියාත්මක වන්නේ Master විසින් ලියාපදිංචි කළ තමන් සතු සර්වර් සහ උපාංග වලට පමණි. පිටස්තර පද්ධති වලට පහර දීම සම්පූර්ණයෙන්ම තහනම් කර ඇත.
Master Passphrase Immunity: AI මොඩලයකට හෝ ස්වයංක්‍රීය Playbook එකකට තමන්ගේම බලතල වැඩි කරගැනීමට කිසිසේත්ම නොහැක.


📜 License & Acknowledgments

F.R.I.D.A.Y. AI
Copyright © 2026 Kaveen Chamod

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
Base Desktop Inspiration: Sathija Balachandra (Jarvis AI core tool concepts).
Face Mesh Assets: MediaPipe (Apache License 2.0).
Neural Voices: Google Gemini Live (Aoede) & Microsoft Edge Neural Speech.
Developed by [Your Name]
F.R.I.D.A.Y. — Tactical Intelligence. Absolute Loyalty. Zero Compromise.
'@ | Out-File -FilePath .\README.md -Encoding utf8