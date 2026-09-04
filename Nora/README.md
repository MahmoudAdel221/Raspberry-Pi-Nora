# NORA — Smart Glasses System
### نظارة ذكية للمكفوفين | Raspberry Pi

---

## 🖥️ System Requirements

**Tested on:** Raspberry Pi 4/5 — Raspberry Pi OS (64-bit, Bookworm)

---

## 📦 Step 1 — System Packages (apt)

Run this once before installing Python dependencies:

```bash
sudo apt update && sudo apt install -y \
  mpg123 \
  chromium-browser \
  chromium-chromedriver \
  portaudio19-dev \
  libatlas-base-dev \
  libjpeg-dev \
  libopenblas-dev \
  libasound2-dev \
  python3-picamera2
```

| Package | Purpose |
|---|---|
| `mpg123` | تشغيل ملفات MP3 (الصوت العربي) |
| `chromium-browser` | متصفح Chromium لـ WhatsApp Web |
| `chromium-chromedriver` | Selenium driver للتحكم في Chromium |
| `portaudio19-dev` | مكتبة الصوت المطلوبة لـ PyAudio |
| `libatlas-base-dev` | تسريع العمليات الحسابية لـ NumPy |
| `libjpeg-dev` | معالجة صور JPEG |
| `libopenblas-dev` | تسريع PyTorch على ARM |
| `libasound2-dev` | مكتبة ALSA للصوت |
| `python3-picamera2` | تشغيل كاميرا الراسبيري باي |

---

## 🐍 Step 2 — Python Virtual Environment

```bash
python3 -m venv N --system-site-packages
source N/bin/activate
```

> `--system-site-packages` مطلوب عشان `picamera2` تشتغل من النظام

---

## 📚 Step 3 — Python Dependencies

```bash
pip install -r requirements.txt
```

> **ملاحظة:** الملف `requirements.txt` بيثبت PyTorch CPU أوتوماتيك قبل ultralytics

---

## ⚙️ Step 4 — Environment Variables

أنشئ ملف `.env` في نفس فولدر الكود:

```bash
nano .env
```

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_API_KEY_2=
GEMINI_API_KEY_3=
EMERGENCY_EMAIL=emergency_contact@example.com
SENDER_EMAIL=your_gmail@gmail.com
SENDER_APP_PASSWORD=your_gmail_app_password
EMERGENCY_PHONE=+962xxxxxxxxx
```

---

## 🚀 Run

```bash
source N/bin/activate
python3 Nora.py
```

---

## 📁 Project Structure

```
project/
├── Nora.py               # Main application
├── requirements.txt      # Python dependencies
├── .env                  # API keys and config (not in git)
├── static                # Contain the image of the background in the web panel
├── trainer.yml           # Face recognition model (auto-generated)
├── face_labels.pkl       # Face labels (auto-generated)
└── whatsapp_session/     # WhatsApp session (auto-generated)
```

---

## 🎙️ Voice Commands (Arabic)

| Command | Function |
|---|---|
| نورا صف المشهد | Scene description |
| نورا الوضع السريع | Fast mode (YOLOv8 only) |
| نورا الوضع التفصيلي | Detailed mode (Gemini) |
| نورا اقرأ | OCR — read text |
| نورا ما العملة | Currency recognition |
| نورا احفظ هذا الشخص باسم ""| Register face |
| نورا وضع الامان | Enable safety mode |
| نورا اغلق الامان | Disable safety mode |
| نورا كم الساعة | Current time |
| نورا ما التاريخ | Current date |
| نورا اعداد الواتساب | WhatsApp session setup |
| نورا حاله طوارئ | Send SOS alert |
