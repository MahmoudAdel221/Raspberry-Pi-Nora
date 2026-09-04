import os
import sys

# ✅ SILENCE SYSTEM ERRORS (JACK/ALSA/LIBCAMERA)
# This redirects the low-level 'stderr' (where JACK prints) to nowhere.
def silence_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stderr.fileno())

# Call it immediately
try:
    silence_stderr()
except Exception:
    pass


import cv2
# speech_recognition imported lazily inside voice_trigger_thread only
from gtts import gTTS
from google import genai
from google.genai import types
from ultralytics import YOLO
import threading
import time
import traceback
from PIL import Image
import numpy as np
import subprocess
import pickle
import platform
import logging
from dotenv import load_dotenv
from datetime import datetime
import requests
import serial
import ctypes
import hashlib
# ============================================================
# ✅ ALSA Error Suppressor (Added Fix 1)
# This silences the "Unknown PCM" warnings from libasound.
# ============================================================
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass



# ============================================================
# ✅ Force CPU-only for PyTorch (CRITICAL for Raspberry Pi)
# This must happen BEFORE any ultralytics/torch import resolves
# to prevent YOLO from probing for CUDA and crashing on ARM.
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Hide all GPUs
os.environ["TORCH_DEVICE"] = "cpu"        # Force CPU
os.environ["OMP_NUM_THREADS"] = "2"       # Limit OpenMP threads (Pi has 4 cores, keep 2 for OS)
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ.setdefault("DISPLAY", ":0")    # Required for cv2.imshow on RPi HDMI

# ============================================================
# ✅ Raspberry Pi Camera Module 3 via picamera2
# picamera2 is the official modern library for Pi cameras.
# cv2.VideoCapture(0) is unreliable with libcamera-based cameras.
# ============================================================
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️  picamera2 not found — falling back to cv2.VideoCapture(0)")

# Selenium (background WhatsApp)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    filename='nora.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================
# ✅ Load API keys from .env
# ============================================================
load_dotenv()
MY_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not MY_API_KEY:
    print("⚠️  GEMINI_API_KEY not found in .env — AI features will fail.")

# ✅ Backup API keys — auto-rotated when primary key hits quota
_GEMINI_KEYS: list = [k for k in [
    MY_API_KEY,
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
] if k]

EMERGENCY_PHONE     = os.getenv("EMERGENCY_PHONE", "")
EMERGENCY_EMAIL     = os.getenv("EMERGENCY_EMAIL", "")
SENDER_EMAIL        = os.getenv("SENDER_EMAIL", "")
SENDER_PW           = os.getenv("SENDER_APP_PASSWORD", "")
CHROME_PROFILE_PATH = os.path.abspath("whatsapp_session")

# ✅ Auto-detect Chrome + chromedriver (works on Bullseye AND Bookworm)
def _find_binary(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

CHROME_BINARY_PATH = _find_binary([
    "/usr/bin/chromium-browser",          # Bullseye
    "/usr/bin/chromium",                   # Bookworm
    "/snap/bin/chromium",                  # Snap
])
CHROMEDRIVER_PATH = _find_binary([
    "/usr/bin/chromedriver",               # Bookworm (chromium-driver)
    "/usr/lib/chromium-browser/chromedriver",  # Bullseye
    "/usr/lib/chromium/chromedriver",
])

# ============================================================
# Global state
# ============================================================
_state_lock         = threading.Lock()   # ✅ protects shared flags
process_request     = False
running             = True
is_processing       = False
is_speaking         = False
safety_mode_enabled = False
current_operating_mode = "fast"   # "fast" | "detailed"
_mode_before_safety = "fast"      # ✅ remember mode before safety was ON
low_battery_mode    = False
_bg_scan_running    = False           # ✅ prevents overlapping YOLO bg scans

yolo_model = None
last_warning_time   = 0
warning_cooldown    = 15.0
last_safety_mode_description_time   = 0
safety_mode_description_interval    = 15.0

# Face recognition
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces_data_file  = "faces_data.pkl"
face_names       = {}
current_face_id  = 0
save_face_request = False
face_to_save_name = ""
face_announce_cooldown    = 10.0
last_face_announce_time   = {}

# Face training (25-frame multi-sample approach)
FACE_TRAINING_FRAMES    = 25
face_training_active    = False
face_training_samples   = []
face_training_ids       = []
face_training_name      = ""
face_training_frame_count = 0

# ✅ Set absolute paths for persistence files to avoid RPi directory issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
faces_data_file = os.path.join(BASE_DIR, "faces_data.pkl")
trainer_file    = os.path.join(BASE_DIR, "trainer.yml")

def load_face_data():
    global face_names, current_face_id
    if os.path.exists(faces_data_file):
        try:
            with open(faces_data_file, 'rb') as f:
                data = pickle.load(f)
                face_names = data.get('names', {})
                if face_names:
                    current_face_id = max(face_names.keys()) + 1
                else:
                    current_face_id = 0
            if os.path.exists(trainer_file):
                recognizer.read(trainer_file)
                print(f"✅ Loaded {len(face_names)} faces.")
        except Exception as e:
            print(f"⚠️ Error loading face data: {e}")

def save_face_data():
    try:
        with open(faces_data_file, 'wb') as f:
            pickle.dump({'names': face_names}, f)
        recognizer.save(trainer_file)
        print("✅ Face data and trainer saved.")
    except Exception as e:
        print(f"⚠️ Error saving face data: {e}")

# Initial Load
load_face_data()


# Arabic object name mapping
ARABIC_NAMES = {
    "person": "شخص", "bicycle": "دراجة", "car": "سيارة", "motorcycle": "دراجة نارية",
    "airplane": "طائرة", "bus": "حافلة", "train": "قطار", "truck": "شاحنة",
    "boat": "قارب", "traffic light": "إشارة مرور", "fire hydrant": "صنبور حريق",
    "stop sign": "لوحة قف", "parking meter": "عداد انتظار", "bench": "مقعد",
    "bird": "عصفور", "cat": "قطة", "dog": "كلب", "horse": "حصان", "sheep": "خروف",
    "cow": "بقرة", "elephant": "فيل", "bear": "دب", "zebra": "حمار وحشي",
    "giraffe": "زرافة", "backpack": "حقيبة ظهر", "umbrella": "مظلة",
    "handbag": "حقيبة يد", "tie": "ربطة عنق", "suitcase": "حقيبة سفر",
    "frisbee": "قرص طائر", "skis": "زلاجات", "snowboard": "لوح تزلج",
    "sports ball": "كرة رياضية", "kite": "طائرة ورقية", "baseball bat": "مضرب بيسبول",
    "baseball glove": "قفاز بيسبول", "skateboard": "لوح تزلج", "surfboard": "لوح ركوب أمواج",
    "tennis racket": "مضرب تنس", "bottle": "زجاجة", "wine glass": "كأس",
    "cup": "كوب", "fork": "شوكة", "knife": "سكين", "spoon": "ملعقة", "bowl": "وعاء",
    "banana": "موزة", "apple": "تفاحة", "sandwich": "ساندوتش", "orange": "برتقالة",
    "broccoli": "بروكلي", "carrot": "جزرة", "hot dog": "هوت دوج", "pizza": "بيتزا",
    "donut": "دونات", "cake": "كعكة", "chair": "كرسي", "couch": "أريكة",
    "potted plant": "نبتة", "bed": "سرير", "dining table": "طاولة طعام",
    "toilet": "مرحاض", "tv": "تلفاز", "laptop": "لابتوب", "mouse": "فأرة",
    "remote": "ريموت", "keyboard": "لوحة مفاتيح", "cell phone": "هاتف",
    "microwave": "ميكروويف", "oven": "فرن", "toaster": "محمصة", "sink": "حوض",
    "refrigerator": "ثلاجة", "book": "كتاب", "clock": "ساعة", "vase": "مزهرية",
    "scissors": "مقص", "teddy bear": "دب لعبة", "hair drier": "مجفف شعر",
    "toothbrush": "فرشاة أسنان"
}


# ============================================================
# ✅ Raspberry Pi Camera Module 3 — capture helper
# ============================================================
_picam2 = None   # global singleton

def init_picamera():
    """Initialize picamera2 singleton. Returns True on success."""
    global _picam2
    if not PICAMERA2_AVAILABLE:
        return False
    try:
        _picam2 = Picamera2()
        # 640×480 is the safest resolution for real-time YOLO on Pi 4
        config = _picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        _picam2.configure(config)
        _picam2.start()
        time.sleep(2)   # warm-up: camera needs ~2 s to stabilise exposure
        print("✅ picamera2 started (Camera Module 3)")
        return True
    except Exception as e:
        print(f"picamera2 init error: {e}")
        _picam2 = None
        return False


def capture_frame():
    """
    Capture one BGR frame.
    Returns: numpy array (H, W, 3) BGR  or  None on failure.
    """
    global _picam2
    if PICAMERA2_AVAILABLE and _picam2 is not None:
        try:
            # picamera2 gives RGB888; convert to BGR for OpenCV
            rgb = _picam2.capture_array()
            #bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            #return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception as e:
            print(f"picamera2 capture error: {e}")
            return None
    return None   # caller falls back to cv2.VideoCapture


# ============================================================
# Speech — subprocess-first (reliable on RPi headless)
# ============================================================
speak_lock = threading.Lock()

# ✅ OPTIMIZATION 1: Cache TTS audio files to avoid repeated network calls
# Store in project dir so cache survives reboots
_tts_cache: dict = {}          # text → file path
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

# ✅ OPTIMIZATION 2: Detect best audio player ONCE at startup
_audio_player: str = ""

def _detect_audio_player() -> str:
    """Find the first working audio player and remember it."""
    for player in ["mpg123", "mpg321", "ffplay", "play"]:
        try:
            result = subprocess.run(["which", player],
                                    capture_output=True, timeout=2)
            if result.returncode == 0:
                print(f"✅ Audio player: {player}")
                return player
        except Exception:
            continue
    print("⚠️  No audio player found. Install mpg123: sudo apt install mpg123")
    return ""


def _play_file(path: str):
    """Play an mp3 file using the pre-detected player."""
    global _audio_player
    if not _audio_player:
        _audio_player = _detect_audio_player()
    if not _audio_player:
        return
    try:
        if _audio_player == "ffplay":
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]
        else:
            cmd = [_audio_player, "-q", path]
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"_play_file error: {e}")


def _get_tts_file(text: str) -> str:
    """Return a cached mp3 path for the text, generating it if needed."""
    if text in _tts_cache and os.path.exists(_tts_cache[text]):
        return _tts_cache[text]
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    path = os.path.join(_CACHE_DIR, f"{h}.mp3")
    if not os.path.exists(path):
        tts = gTTS(text=text, lang="ar", slow=False)
        tts.save(path)
    _tts_cache[text] = path
    return path


def _prewarm_common_phrases():
    """Pre-generate TTS for the most frequent replies at startup (background)."""
    phrases = [
        "نعم", "جاري التنفيذ.", "تم.",
        "لا أرى شيئاً مهماً.",
        "مرحباً، نورا معك الآن، جاهزة لمساعدتك.",
        "تم تفعيل وضع الأمان.",
        "تم إيقاف وضع الأمان.",
        "تم تفعيل الوضع السريع.",
        "تم تفعيل الوضع التفصيلي.",
        "تم تفعيل وضع توفير الطاقة.",
        "جاري إرسال رسالة استغاثة…",
    ]
    for p in phrases:
        try:
            _get_tts_file(p)
        except Exception:
            pass

def speak_arabic(text, priority=False):
    global is_speaking
    if is_speaking and not priority:
        return

    def run_speak():
        global is_speaking
        with speak_lock:
            try:
                is_speaking = True
                print(f"Assistant: {text}")
                voice_file = _get_tts_file(text)
                _play_file(voice_file)
            except Exception as e:
                print(f"speak_arabic error: {e}")
            finally:
                time.sleep(0.2)
                is_speaking = False

    threading.Thread(target=run_speak, daemon=True).start()


def _generate_beep_file():
    """Generate a short beep mp3 as fallback alarm."""
    beep_path = os.path.join(BASE_DIR, "_alarm_beep.mp3")
    if os.path.exists(beep_path):
        return beep_path
    try:
        # Generate beep using gTTS as a workaround (short text = short audio)
        tts = gTTS(text="تنبيه! تنبيه!", lang="ar", slow=False)
        tts.save(beep_path)
        return beep_path
    except:
        return ""

def play_alarm_sound():
    """Play alarm file via audio player in a separate thread."""
    alarm_file = os.path.join(BASE_DIR, "alarm-warning-beeps-ra-music-1-00-02.mp3")
    if not os.path.exists(alarm_file):
        # ✅ FIX: Generate beep fallback instead of silently failing
        alarm_file = _generate_beep_file()
        if not alarm_file:
            print("⚠️  No alarm file and couldn't generate beep.")
            return

    def run_alarm():
        try:
            _play_file(alarm_file)
        except Exception as e:
            print(f"Alarm error: {e}")

    threading.Thread(target=run_alarm, daemon=True).start()


# ============================================================
# Time & Date helpers
# ============================================================
def get_time_response():
    now = datetime.now()
    hour, minute = now.hour, now.minute
    period = "صباحاً" if hour < 12 else "مساءً"
    hour_12 = hour if hour <= 12 else hour - 12
    if hour_12 == 0:
        hour_12 = 12
    if minute == 0:
        return f"الساعة {hour_12} {period} تماماً"
    elif minute == 30:
        return f"الساعة {hour_12} والنصف {period}"
    elif minute == 15:
        return f"الساعة {hour_12} والربع {period}"
    elif minute == 45:
        return f"الساعة {hour_12 + 1} إلا ربع {period}"
    else:
        return f"الساعة {hour_12} و{minute} دقيقة {period}"


def get_date_response():
    now = datetime.now()
    days_ar   = ["الاثنين","الثلاثاء","الأربعاء","الخميس","الجمعة","السبت","الأحد"]
    months_ar = ["يناير","فبراير","مارس","أبريل","مايو","يونيو",
                 "يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر"]
    return f"اليوم {days_ar[now.weekday()]}، {now.day} {months_ar[now.month-1]} {now.year}"


# ============================================================
# GPS via 4G Shield serial
# ============================================================
def get_real_gps_location():
    """Try common RPi serial ports for NMEA GPS data."""
    for port in ['/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3', '/dev/serial0', '/dev/ttyAMA0']:
        try:
            ser = serial.Serial(port, 115200, timeout=2)
            ser.write(b"AT+CGPS=1\r\n")
            time.sleep(1)
            ser.write(b"AT+CGPSINFO\r\n")
            time.sleep(1)
            response = ser.read_all().decode('utf-8', errors='ignore')
            ser.close()
            if "+CGPSINFO:" in response:
                data = response.split("+CGPSINFO:")[1].split("\r")[0].strip()
                parts = data.split(",")
                if len(parts) >= 4 and parts[0] and parts[2]:
                    lat_raw, lat_dir, lon_raw, lon_dir = parts[0], parts[1], parts[2], parts[3]
                    lat = float(lat_raw[:2])  + float(lat_raw[2:]) / 60.0
                    if lat_dir == 'S': lat = -lat
                    lon = float(lon_raw[:3]) + float(lon_raw[3:]) / 60.0
                    if lon_dir == 'W': lon = -lon
                    return f"{lat},{lon}"
        except Exception:
            continue
    return None


# ============================================================
# Email helpers
# ============================================================
def send_sos_email(maps_link):
    import smtplib, ssl
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    if not SENDER_EMAIL or not SENDER_PW:
        print("⚠️  SOS email credentials missing in .env")
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To']   = EMERGENCY_EMAIL
        msg['Subject'] = "🚨 نداء استغاثة عاجل من مستخدم النظارة الذكية!"
        body = f"مستخدم النظارة الذكية في حالة خطر.\n\nموقعه: {maps_link}\n\nأُرسل تلقائياً."
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as srv:
            srv.login(SENDER_EMAIL, SENDER_PW)
            srv.sendmail(SENDER_EMAIL, EMERGENCY_EMAIL, msg.as_string())
        print("✅ SOS email sent.")
        return True
    except Exception as e:
        print(f"SOS email error: {e}")
        return False


def send_qr_email(qr_image_path):
    import smtplib, ssl
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage
    if not SENDER_EMAIL or not SENDER_PW:
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To']   = EMERGENCY_EMAIL
        msg['Subject'] = "📱 رمز QR لتسجيل الدخول للواتساب - النظارة الذكية"
        msg.attach(MIMEText("امسح الرمز بهاتفك عبر واتساب لتفعيل خدمة الطوارئ.", 'plain', 'utf-8'))
        with open(qr_image_path, 'rb') as f:
            img = MIMEImage(f.read(), name=os.path.basename(qr_image_path))
        msg.attach(img)
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as srv:
            srv.login(SENDER_EMAIL, SENDER_PW)
            srv.sendmail(SENDER_EMAIL, EMERGENCY_EMAIL, msg.as_string())
        print("✅ QR email sent.")
        return True
    except Exception as e:
        print(f"QR email error: {e}")
        return False


# ============================================================
# WhatsApp via Selenium (headless Chromium on RPi)
# ============================================================
whatsapp_busy = False


def _make_chrome_options():
    options = Options()
    if os.path.exists(CHROME_BINARY_PATH):
        options.binary_location = CHROME_BINARY_PATH
    options.add_argument("--headless=new")
    options.add_argument(f"--user-data-dir={CHROME_PROFILE_PATH}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--remote-allow-origins=*")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    # ✅ User-Agent يطابق Chromium 147 على الراسبيري باي
    options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux aarch64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    )
    return options


def _get_driver():
    """
    ✅ On RPi always use the system chromedriver to avoid webdriver-manager
    hitting the network and downloading an incompatible (x86) binary.
    """
    options = _make_chrome_options()
    if os.path.exists(CHROMEDRIVER_PATH):
        d = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)
    else:
        # Fallback: let Selenium find chromedriver on PATH
        d = webdriver.Chrome(options=options)
    # ✅ إخفاء علامات الأتمتة عن واتساب
    d.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
            window.chrome = { runtime: {} };
        """
    })
    return d


def _clear_chrome_lock():
    for lf in ["SingletonLock", "SingletonSocket", "SingletonCookie"]:
        p = os.path.join(CHROME_PROFILE_PATH, lf)
        if os.path.exists(p):
            try: os.remove(p)
            except: pass


def init_whatsapp_session():
    global whatsapp_busy
    if whatsapp_busy:
        return "يتم استخدام الواتساب الآن، يرجى الانتظار."
    whatsapp_busy = True
    driver = None
    try:
        os.makedirs(CHROME_PROFILE_PATH, exist_ok=True)
        _clear_chrome_lock()
        driver = _get_driver()
        driver.get("https://web.whatsapp.com")
        speak_arabic("جاري تجهيز واتساب. يرجى الانتظار.")

        # Already logged in?
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//div[@id='pane-side']"))
            )
            speak_arabic("الواتساب بالفعل مربوط وجاهز.")
            driver.quit()
            whatsapp_busy = False
            return "الواتساب مربوط مسبقاً."
        except:
            pass

        speak_arabic("جاري سحب صورة الرمز وإرسالها للإيميل.")
        qr_path    = "/tmp/whatsapp_qr.png"
        qr_captured = False

        for attempt in range(3):
            for xpath in [
                "//canvas[@aria-label='Scan me!']",
                "//div[@data-ref]//canvas",
                "//canvas",
                "//div[@data-testid='qrcode']",
            ]:
                try:
                    el = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, xpath))
                    )
                    el.screenshot(qr_path)
                    qr_captured = True
                    break
                except:
                    continue
            if qr_captured:
                break
            time.sleep(5)

        if qr_captured:
            if send_qr_email(qr_path):
                speak_arabic("تم إرسال رمز الواتساب لإيميلك. امسحه بهاتفك خلال دقيقة.")
            else:
                speak_arabic("فشل الإرسال، تأكد من بيانات الإيميل في ملف .env")
        else:
            speak_arabic("فشلت في العثور على رمز الربط. تأكد من الإنترنت.")
            driver.quit()
            whatsapp_busy = False
            return "فشل التقاط الرمز."

        try:
            WebDriverWait(driver, 120).until(
                EC.presence_of_element_located((By.XPATH, "//div[@id='pane-side']"))
            )
            time.sleep(15)   # let session cookies persist to disk
            speak_arabic("تم ربط الواتساب بنجاح، جاهز لحالات الطوارئ.")
        except:
            speak_arabic("انتهى وقت المسح. أعد المحاولة.")

        driver.quit()
        whatsapp_busy = False
        return "اكتملت عملية الربط."
    except Exception:
        if driver:
            try: driver.quit()
            except: pass
        whatsapp_busy = False
        traceback.print_exc()
        return "حدث خطأ أثناء الربط."


def send_sos_whatsapp_selenium():
    global whatsapp_busy
    if whatsapp_busy:
        return "يرجى الانتظار حتى ينتهي إعداد الواتساب."
    whatsapp_busy = True
    driver = None
    try:
        # ✅ Clean up any stuck processes
        subprocess.run(["pkill", "-f", "chromedriver"], capture_output=True)
        time.sleep(1)
        _clear_chrome_lock()

        loc       = get_real_gps_location()
        maps_link = f"https://www.google.com/maps?q={loc}" if loc else "موقع غير متاح"
        if not loc:
            try:
                res  = requests.get('https://ipinfo.io/json', timeout=5).json()
                loc2 = res.get('loc', '')
                maps_link = f"https://www.google.com/maps?q={loc2}" if loc2 else "تعذر تحديد الموقع"
            except:
                maps_link = "تعذر تحديد الموقع"

        # ✅ Send email immediately in background
        threading.Thread(target=send_sos_email, args=(maps_link,), daemon=True).start()

        msg    = f"نداء استغاثة عاجل!\n\nاحتاج للمساعدة. موقعي: {maps_link}"
        try:
            driver = _get_driver()
            # ✅ Pi needs a LOT of time for Chromium
            time.sleep(15) 

            encoded_msg = requests.utils.quote(msg)
            url = f"https://web.whatsapp.com/send?phone={EMERGENCY_PHONE}&text={encoded_msg}"
            driver.get(url)

            # ✅ Wait up to 90 seconds for slow RPi internet
            wait = WebDriverWait(driver, 90)
            try:
                # ✅ الرسالة محطوطة تلقائي في الـ URL (?text=...)
                # بنستنى الـ input box يظهر، نعمله focus، ونضغط Enter
                from selenium.webdriver.common.keys import Keys
                
                input_box = wait.until(EC.presence_of_element_located((
                    By.XPATH,
                    "//div[@contenteditable='true'][@data-tab='10'] | "
                    "//footer//div[@role='textbox']"
                )))
                time.sleep(2)
                input_box.click()
                time.sleep(1)
                input_box.send_keys(Keys.ENTER)
                time.sleep(15)  # ✅ وقت كافي عشان واتساب يوصّل الرسالة
                print("✅ SOS WhatsApp message sent.")
            except Exception as inner:
                # Fallback: نجرب Enter عام
                try:
                    from selenium.webdriver.common.action_chains import ActionChains
                    ActionChains(driver).send_keys(Keys.ENTER).perform()
                    time.sleep(10)
                    print("✅ SOS WhatsApp sent via ENTER fallback.")
                except Exception as fallback_err:
                    print(f"SOS WhatsApp inner error: {inner}")
                    print(f"Fallback also failed: {fallback_err}")

            time.sleep(5)
        except Exception as wa_err:
            print(f"⚠️ WhatsApp failed (email still sent): {wa_err}")
        finally:
            if driver:
                try: driver.quit()
                except: pass

        whatsapp_busy = False
        return "تم إرسال رسالة الاستغاثة بنجاح عبر الإيميل وجاري الإرسال عبر الواتساب."
    except Exception as e:
        if driver:
            try: driver.quit()
            except: pass
        whatsapp_busy = False
        traceback.print_exc()
        return "حدث خطأ أثناء إرسال الاستغاثة."


# ============================================================
# Gemini helper + Currency + OCR + Scene description
# ============================================================
_gemini_last_call     = 0.0
_GEMINI_MIN_GAP       = 4.0   # seconds between calls (free tier ~15 rpm)
_gemini_key_index     = 0     # ✅ current active key index
_gemini_key_blocked: dict = {}  # key → blocked_until timestamp

def _gemini_call(contents, retries=2):
    """Call Gemini with key-rotation on 429 + rate-limiting."""
    global _gemini_last_call, _gemini_key_index

    if not _GEMINI_KEYS:
        print("⚠️  No Gemini API keys configured.")
        return ""

    now = time.time()
    # Try each key in order, skipping blocked ones
    for attempt in range(len(_GEMINI_KEYS)):
        idx = (_gemini_key_index + attempt) % len(_GEMINI_KEYS)
        key = _GEMINI_KEYS[idx]

        # Skip if this key is still blocked
        if _gemini_key_blocked.get(key, 0) > now:
            mins = int((_gemini_key_blocked[key] - now) // 60)
            print(f"Key {idx+1} blocked for {mins} more min, trying next")
            continue

        # Rate-limit: enforce minimum gap
        gap = now - _gemini_last_call
        if gap < _GEMINI_MIN_GAP:
            time.sleep(_GEMINI_MIN_GAP - gap)

        try:
            _gemini_last_call = time.time()
            client = genai.Client(api_key=key)
            resp   = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents
            )
            _gemini_key_index = idx  # remember the working key
            return resp.text

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                # Daily quota exhausted — block this key for 60 min and try next
                _gemini_key_blocked[key] = time.time() + 3600
                key_label = f"key {idx+1}/{len(_GEMINI_KEYS)}"
                print(f"⚠️  Gemini {key_label} quota exhausted — switching to next key")
                continue   # immediately try next key
            else:
                print(f"Gemini error: {e}")
                return ""

    # All keys exhausted
    print("❌ All Gemini API keys are quota-exhausted. Add more keys to .env")
    return ""


def recognize_currency(frame):
    if low_battery_mode:
        return "ميزة التعرف على العملات متوقفة لحفظ البطارية."
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text = _gemini_call([
            "أنت مساعد لشخص مكفوف. استخرج واقرأ قيمة العملة النقدية في هذه الصورة بدقة "
            "باللغة العربية بكلمتين أو ثلاث (مثل: خمسون جنيهاً مصرياً). لا تضف مقدمات. "
            "إذا لم تجد عملة، قل: لا توجد عملة نقدية واضحة.",
            pil_img
        ]).strip()
        return text if len(text) > 2 else "لا توجد عملة نقدية واضحة."
    except Exception as e:
        print(f"Currency error: {e}")
        return "حدث خطأ في التعرف على العملة."


# ============================================================
# Gemini scene description
# ============================================================
def get_gemini_response(pil_img, force=False):
    """force=True يتجاهل وضع التشغيل الحالي ويستدعي Gemini دائماً (للطلبات اليدوية)."""
    if not force and (current_operating_mode == "fast" or low_battery_mode):
        return ""
    text = _gemini_call([
        "أنت مساعد لشخص مكفوف. صف هذا المشهد بوضوح وبالعربية بجملتين أو ثلاث. "
        "اذكر أهم الأشياء والأشخاص وأماكنهم. لا تقل أنا أصف أو ما شابه، ابدأ مباشرة بالوصف.",
        pil_img
    ])
    return text


# ============================================================
# OCR via Gemini
# ============================================================


def read_text_from_frame(frame):
    if low_battery_mode:
        return "ميزة قراءة النصوص متوقفة لحفظ البطارية."
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text = _gemini_call([
            "أنت مساعد لشخص مكفوف. استخرج واقرأ النص الموجود في هذه الصورة بدقة. "
            "اكتب النص الواضح فقط، ولا تضف مقدمات. إذا لم يكن هناك نص قل: لا يوجد نص واضح أمامي.",
            pil_img
        ]).strip()
        return f"النص الذي أمامي هو: {text}" if len(text) > 2 else "لم أجد نصاً واضحاً."
    except Exception as e:
        print(f"OCR error: {e}")
        return "حدث خطأ في قراءة النص."


# ============================================================
# Voice trigger thread
# ============================================================
def voice_trigger_thread():
    global process_request, running, safety_mode_enabled, current_operating_mode
    global is_speaking, save_face_request, face_to_save_name, low_battery_mode
    global last_warning_time

    try:
        import speech_recognition as sr
        import pyaudio
        pa = pyaudio.PyAudio()
        default_index = None
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                if info['name'] == 'default' or default_index is None:
                    default_index = i
                    if info['name'] == 'default':
                        break
        pa.terminate()
        mic = sr.Microphone(device_index=default_index, sample_rate=16000)
    except Exception as e:
        print(f"⚠️  Voice init failed: {e} — voice commands disabled.")
        return

    print("✅ Voice thread started.")
    r = sr.Recognizer()

    # ✅ OPTIMIZATION 3: Fast response but still captures full sentences
    r.energy_threshold = 250          # حساسية عالية
    r.dynamic_energy_threshold = True
    r.pause_threshold = 2           # ✅ FIX: 1.2 ثانية صمت = رد سريع بدون قطع
    r.non_speaking_duration = 0.5     # نصف ثانية فقط
    r.operation_timeout = None        # لا timeout على العملية
    r.phrase_threshold = 0.6

    # ✅ OPTIMIZATION 4: Calibrate ONCE at startup (not every loop)
    print("🎙️  Calibrating microphone noise level...")
    try:
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=2)
        print(f"✅ Mic calibrated. Energy threshold: {r.energy_threshold:.0f}")
    except Exception as e:
        print(f"⚠️  Mic calibration failed: {e}")

    # Recalibrate every 5 minutes in background
    _last_calibration = time.time()
    _RECAL_INTERVAL   = 300  # seconds

    while running:
        if is_speaking:
            time.sleep(0.05)
            continue

        # Periodic recalibration (non-blocking)
        if time.time() - _last_calibration > _RECAL_INTERVAL:
            try:
                with mic as source:
                    r.adjust_for_ambient_noise(source, duration=1)
                _last_calibration = time.time()
            except Exception:
                pass

        try:
            with mic as source:
                # timeout=5: ينتظر 5 ثواني لصوت جديد (أسرع)
                # phrase_time_limit=12: يسمح بجمل طويلة بدون قطع
                audio = r.listen(source, timeout=5, phrase_time_limit=12)
            if is_speaking:
                continue
            command = r.recognize_google(audio, language="ar-EG")
            print(f"Recognized: {command}")
            logging.info(f"Voice: {command}")

            if "نورا" not in command and "نوره" not in command and "نور" not in command:
                continue

            extracted = ""
            if "نورا" in command:
                extracted = command.split("نورا", 1)[1].strip()
            elif "نوره" in command:
                extracted = command.split("نوره", 1)[1].strip()
            elif "نور" in command:
                extracted = command.split("نور", 1)[1].strip()

            if len(extracted) < 2:
                speak_arabic("نعم", priority=True)
                # انتظر انتهاء الكلام
                while is_speaking:
                    time.sleep(0.05)
                try:
                    with mic as source:
                        audio   = r.listen(source, timeout=5, phrase_time_limit=12)
                    command = r.recognize_google(audio, language="ar-EG")
                    print(f"Command: {command}")
                except Exception:
                    continue
            else:
                command = extracted

            # ✅ Helper to ensure we don't activate if "stop" words are present
            has_stop = any(s in command for s in ["اغلق", "وقف", "الغي", "إلغاء", "عطل", "ايقاف", "إيقاف", "بطل", "اطفي", "أغلق"])

            # ── Execute command ─
            if any(w in command for w in ["اغلق البرنامج", "اوقف البرنامج", "قفل البرنامج", "اطفي البرنامج", "إغلاق البرنامج", "وداعا"]):
                speak_arabic("جاري إغلاق البرنامج. وداعاً.", priority=True)
                time.sleep(2)
                running = False
                os._exit(0)

            # ── Time / Date ───────────────────────────────────────
            elif any(w in command for w in ["كم الساعة", "ما الوقت", "الساعة كام", "الوقت", "كام الساعه", "الساعه كام", "قولي الوقت", "الساعة","الساعه كام"]):
                speak_arabic(get_time_response(), priority=True)

            elif any(w in command for w in ["ما التاريخ", "ما اليوم", "اليوم إيه", "التاريخ", "اليوم ايه", "قولي التاريخ", "تاريخ النهارده","التاريخ ايه","النهارده كام فى الشهر", "النهارده ايه"]):
                speak_arabic(get_date_response(), priority=True)

            # ══════════════════════════════════════════════════════
            # ✅ STRICT PRIORITY: OFF Commands First
            # ══════════════════════════════════════════════════════
            
            # 1. STOP SAFETY MODE → restore previous mode
            elif any(w in command for w in ["اغلق الامان", "أغلق الامان", "اوقف الامان", "وقف الامان", "إلغاء الامان", "الغاء الامان", "عطل الامان", "اطفي الامان", "بطل الامان", "اغلق وضع الامان", "وقف وضع الامان", "ايقاف الامان", "إيقاف الامان","اوقف الوضع الام","اوقف وضع الام","وقف وضع الام","وقف الوضع الام","اغلق وضع الام","اغلق الوضع الام"]):
                safety_mode_enabled = False
                current_operating_mode = _mode_before_safety
                mode_name = "التفصيلي" if _mode_before_safety == "detailed" else "السريع"
                speak_arabic(f"تم إيقاف وضع الأمان. رجعت للوضع {mode_name}.", priority=True)

            # 2. STOP BATTERY MODE
            elif any(w in command for w in [
                "اغلق توفير الطاقة", "اغلق توفير الطاقه", "وقف توفير الطاقة", "وقف توفير الطاقه",
                "إلغاء توفير الطاقة", "الغاء توفير الطاقة", "الغاء توفير الطاقه",
                "عطل توفير الطاقة", "عطل توفير الطاقه",
                "اغلق وضع البطارية", "اغلق وضع البطاريه", "وقف وضع البطارية", "وقف وضع البطاريه",
                "الغي وضع توفير الطاقة", "الغي وضع توفير الطاقه",
                "ارجع للوضع العادي", "ارجع للوضع العادى",
                "وقف التوفير", "اوقف التوفير",
                "إلغاء التوفير", "الغاء التوفير", "الغي التوفير",
                "اغلق التوفير", "اطفي التوفير", "بطل التوفير",
                "الغي توفير", "الغاء توفير",
                "اغلق وضع توفير", "وقف وضع توفير", "اوقف وضع توفير",
                "اوقف توفير الطاقه", "اوقف توفير الطاقة",
                "الغي الطاقه", "الغي الطاقة",
                "اغلق الطاقه", "اغلق الطاقة",
                "وقف توفير", "اوقف توفير",
            ]):
                low_battery_mode       = False
                current_operating_mode = "detailed"
                speak_arabic("تم إلغاء وضع توفير الطاقة.", priority=True)

            # 3. STOP FAST MODE
            elif any(w in command for w in ["اغلق الوضع السريع", "وقف الوضع السريع", "الغي الوضع السريع", "عطل الوضع السريع", "ايقاف الوضع السريع", "إيقاف الوضع السريع","اوقف الوضع السري","اوقف  وضع السري","اغلق الوضع السري","اغلق وضع السرى"]):
                low_battery_mode       = False
                current_operating_mode = "detailed"
                speak_arabic("تم إيقاف الوضع السريع والتحويل للوضع التفصيلي.", priority=True)

            # 4. STOP DETAILED MODE
            elif any(w in command for w in ["اغلق الوضع التفصيلي", "وقف الوضع التفصيلي", "الغي الوضع التفصيلي", "عطل الوضع التفصيلي", "ايقاف الوضع التفصيلي","اغلق الوضع التفص","اغلق وضع التفص","اغلق الوضع التفصي","اغلق الوضع التفصيل"]):
                current_operating_mode = "fast"
                speak_arabic("تم إيقاف الوضع التفصيلي والتحويل للوضع السريع.", priority=True)

            # ══════════════════════════════════════════════════════
            # ✅ ON Commands (Only if OFF wasn't detected above)
            # ══════════════════════════════════════════════════════

            # START SAFETY MODE → remember current mode first
            elif not has_stop and any(w in command for w in ["وضع الامان", "الوضع الامن", "شغل الامان", "فعل الامان",
             "تفعيل الامان", "شغل وضع الامان", "وضع الحماية", "الوضع الامني","وضع الام","الوضع الام"]):
                _mode_before_safety = current_operating_mode  # ✅ save current
                speak_arabic("تم تفعيل وضع الأمان.", priority=True)
                time.sleep(2.5)
                safety_mode_enabled = True
                last_warning_time   = time.time()

            # START BATTERY MODE
            elif not has_stop and any(w in command for w in ["توفير الطاقة", "توفير الطاقه", "وضع البطارية",
             "وضع البطاريه", "وفر الطاقة", "شغل توفير الطاقة",
              "فعل توفير الطاقة", "بطاريه منخفضه","توفير البطاريه",
             "توفير البطارية","البطاريه","البطارية","توفير"]):
                low_battery_mode       = True
                current_operating_mode = "fast"
                speak_arabic("تم تفعيل وضع توفير الطاقة.", priority=True)

            # START FAST MODE
            elif not has_stop and any(w in command for w in ["الوضع السريع", "وضع سريع", "شغل السريع", "فعل السريع",
             "الوضع المختصر","الوضع السر","الوضع السرى"," وضع سر","وضع سرى","الوضع الس"]):
                current_operating_mode = "fast"
                speak_arabic("تم تفعيل الوضع السريع.", priority=True)

            # START DETAILED MODE
            elif not has_stop and any(w in command for w in ["الوضع التفصيلي", "الوضع التفصيلى", "وضع تفصيلي",
             "شغل التفصيلي", "فعل التفصيلي", "الوضع المفصل"]):
                low_battery_mode       = False
                current_operating_mode = "detailed"
                speak_arabic("تم تفعيل الوضع التفصيلي.", priority=True)

            # ── Save face ─────────────────────────────────────────
            elif any(p in command for p in [
                "احفظلي هذا الشخص","احفظلي هذا الوجه","احفظ هذا الشخص","احفظ هذا الوجه",
                "سجل هذا الشخص","سجل هذا الوجه","حفظ هذا الشخص","حفظ هذا الوجه",
                "تذكر هذا الشخص","تذكر هذا الوجه","سجلي هذا الشخص","سجلي هذا الوجه"
            ]):
                name = "شخص غير معروف"
                save_phrases = [
                    "احفظلي هذا الشخص باسم","احفظلي هذا الوجه باسم",
                    "احفظ هذا الشخص باسم","احفظ هذا الوجه باسم",
                    "احفظلي هذا الشخص بإسم","احفظلي هذا الوجه بإسم",
                    "احفظ هذا الشخص بإسم","احفظ هذا الوجه بإسم",
                    "احفظلي هذا الشخص اسمه","احفظلي هذا الوجه اسمه",
                    "احفظ هذا الشخص اسمه","احفظ هذا الوجه اسمه",
                    "سجل هذا الشخص باسم","سجل هذا الوجه باسم",
                    "حفظ هذا الشخص باسم","حفظ هذا الوجه باسم",
                    "تذكر هذا الشخص باسم","تذكر هذا الوجه باسم",
                ]
                for phrase in save_phrases:
                    if phrase in command:
                        extracted_name = command[command.find(phrase) + len(phrase):].strip()
                        if extracted_name:
                            name = extracted_name
                        break
                face_to_save_name = name if name != "شخص غير معروف" else f"شخص {current_face_id + 1}"
                speak_arabic(f"جاري حفظ الوجه باسم {face_to_save_name}. انظر للكاميرا بثبات.", priority=True)
                save_face_request = True

            # ── OCR ───────────────────────────────────────────────
            elif any(w in command for w in ["اقرأ","اقرا","اقرأ النص","اقرا النص","فيه نص",
                                            "في نص","ماذا مكتوب","إيه المكتوب","ايه المكتوب",
                                            "مكتوب ايه","مكتوب إيه","اقرالي","اقرا النص ده",
                                            "فيه كلام","ايه الكلام ده","اقرأى","اقرى","اقراى",
                                            "اقرأى النص","اقرى النص","اقراى النص"]):
                with _state_lock:
                    process_request = "ocr"

            # ── Currency ──────────────────────────────────────────
            elif any(w in command for w in ["عملة","عمله","نقود","فلوس","مصاري","كم هذا",
                                            "هذه العملة","هذه العمله","كام دول","كام دي",
                                            "ورقة بكام","ده بكام","دي بكام","فئه كام","فئة كام","ما العم","العم"]):
                with _state_lock:
                    process_request = "currency"

            # ── SOS ───────────────────────────────────────────────
            elif any(w in command for w in ["طوارئ","طوار","حالة طوارئ","انقذوني","مساعدة",
                                            "مساعده","النجدة","النجده","الحقوني","في خطر",
                                            "استغاثة","استغاثه","انقذني","ابعت استغاثة",
                                            "ارسل استغاثة","فيه خطر"]):
                with _state_lock:
                    process_request = "sos"

            # ── WhatsApp init ─────────────────────────────────────
            elif (any(w in command for w in ["ربط","إعداد","اعداد","تجهيز","تفعيل","اعمل","هيئ"]) and
                  any(w in command for w in ["واتساب","وتساب","الواتس","الوتس","واتس","الواتساب"])):
                with _state_lock:
                    process_request = "whatsapp_init"

            # ── Describe scene ────────────────────────────────────
            elif any(w in command for w in ["صف","وصفلي","وصف","صفلي","بتشوف","ايه اللي",
                                            "إيه اللي","إيه قدامك","ايه قدامك","في قدامك",
                                            "شايف ايه","شايف إيه","فيه ايه","فيه إيه",
                                            "ماذا ترى","وصف المشهد","اوصفلي","المشهد",
                                            "صيف المشهد","صيف المش","صف المش","صف المشه",
                                            "صيف المشه","سيف المشهد","سيف المش","يوسف المش",
                                            "يوسف المشهد","يوسف المشه","يوسف","يوسف الم","اوصفى","اوصفى المشهد"]):
                if not is_processing:
                    with _state_lock:
                        process_request = True


        except sr.UnknownValueError:
            continue
        except sr.WaitTimeoutError:
            continue   # ✅ normal silence — not an error
        except Exception as e:
            err = str(e)
            if "listening timed out" in err.lower():
                continue  # ✅ silence timeout — ignore silently
            print(f"Voice error: {e}")
            time.sleep(1)


# ============================================================
# Face training — 25-frame collection
# ============================================================
def collect_face_frame(frame):
    global face_training_active, face_training_samples, face_training_ids
    global face_training_name, face_training_frame_count
    global current_face_id, save_face_request

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ✅ More lenient params so face is detected even at angles / distance
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20)
    )

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi          = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        face_training_samples.extend([roi, cv2.flip(roi, 1)])
        face_training_ids.extend([current_face_id, current_face_id])
        face_training_frame_count += 1

        # ✅ Speak progress every 5 frames so user knows it's working
        if face_training_frame_count % 5 == 0:
            remaining = FACE_TRAINING_FRAMES - face_training_frame_count
            if remaining > 0:
                speak_arabic(f"جيد، {remaining} ثانية.")

        cv2.putText(frame, f"Training: {face_training_frame_count}/{FACE_TRAINING_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if face_training_frame_count >= FACE_TRAINING_FRAMES:
            try:
                # Prepare training data
                ids_arr = np.array(face_training_ids)
                
                if os.path.exists(trainer_file):
                    recognizer.update(face_training_samples, ids_arr)
                else:
                    recognizer.train(face_training_samples, ids_arr)
                
                # Update metadata BEFORE saving
                face_names[current_face_id] = face_training_name
                save_face_data()
                
                # ✅ FIX: Reload recognizer so it can match the face immediately
                recognizer.read(trainer_file)
                print(f"✅ Recognizer reloaded with {len(face_names)} faces")
                
                current_face_id += 1
                speak_arabic(f"تم حفظ الوجه بنجاح باسم {face_training_name}.", priority=True)
            except Exception as e:
                speak_arabic("حدث خطأ أثناء حفظ الوجه.", priority=True)
                print(f"❌ Training error: {e}")
            finally:
                face_training_active      = False
                face_training_samples     = []
                face_training_ids         = []
                face_training_name        = ""
                face_training_frame_count = 0
                save_face_request         = False
    else:
        cv2.putText(frame, "No face detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


# ============================================================
# Scene processing
# ============================================================
def process_frame_logic(frame, is_manual_request=False):
    global is_processing, last_warning_time, safety_mode_enabled
    global last_safety_mode_description_time, last_face_announce_time

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ✅ FIX: Match detection params with training for consistency
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

    face_map = []
    if len(faces) > 0 and os.path.exists(trainer_file):
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            try:
                id_, conf = recognizer.predict(roi)
                if conf < 80:
                    face_map.append((x, y, w, h, face_names.get(id_, "شخص غير معروف"), True))
                else:
                    face_map.append((x, y, w, h, "شخص غير معروف", False))
            except:
                face_map.append((x, y, w, h, "شخص غير معروف", False))
    else:
        for (x, y, w, h) in faces:
            face_map.append((x, y, w, h, "شخص غير معروف", False))

    # ── YOLO (CPU-only on Pi) ────────────────────────────────
    results = yolo_model.predict(frame, conf=0.45, verbose=False, device="cpu")

    detected_objects      = []
    warning_messages      = []
    final_persons_description = []
    matched_faces         = [False] * len(face_map)

    fh_frame, fw_frame, _ = frame.shape
    left_t  = fw_frame // 3
    right_t = 2 * fw_frame // 3

    for r in results:
        for box in r.boxes:
            cls_en = yolo_model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            direction = "أمامك" if left_t <= cx <= right_t else ("على يسارك" if cx < left_t else "على يمينك")
            bw = x2 - x1

            if cls_en == "person":
                # ✅ Face matching — use face rec name if known
                matched_name    = "شخص غير معروف"
                is_known        = False
                for i, (fx, fy, fw, fh_f, fname, fknown) in enumerate(face_map):
                    if matched_faces[i]: continue
                    if (x1-50 <= fx+fw//2 <= x2+50) and (y1-50 <= fy+fh_f//2 <= y2+50):
                        matched_faces[i] = True
                        matched_name     = fname
                        is_known         = fknown
                        break
                label = matched_name if is_known else "شخص غير معروف"
                final_persons_description.append(f"{label} {direction}")
                # ✅ Safety warning for person
                if safety_mode_enabled and bw > fw_frame * 0.35:
                    now = time.time()
                    if now - last_warning_time > warning_cooldown:
                        warning_messages.append(f"{label} {direction}")
                        last_warning_time = now
                continue   # ✅ Skip detected_objects — person handled by face rec only

            # ── Non-person objects ──
            cls_ar = ARABIC_NAMES.get(cls_en, cls_en)
            detected_objects.append(f"{cls_ar} {direction}")
            if safety_mode_enabled and bw > fw_frame * 0.35:
                now = time.time()
                if now - last_warning_time > warning_cooldown:
                    warning_messages.append(f"{cls_ar} {direction}")
                    last_warning_time = now

    # ✅ Add known faces that YOLO didn't detect as "person"
    for i, (fx, fy, fw, fh_f, fname, fknown) in enumerate(face_map):
        if not matched_faces[i] and fknown:
            cx = fx + fw // 2
            direction = "أمامك" if left_t <= cx <= right_t else ("على يسارك" if cx < left_t else "على يمينك")
            final_persons_description.append(f"{fname} {direction}")

    # ✅ Separate known faces from unknown persons
    known_faces_desc   = [d for d in final_persons_description if "غير معروف" not in d]
    unknown_person_count = len(final_persons_description) - len(known_faces_desc)

    if is_manual_request:
        is_processing = True
        try:
            if current_operating_mode == "fast":
                # ══ Fast mode: YOLO objects + face names, NO Gemini ══
                parts = []
                # ✅ Build people description with "و" between names and unknown count
                people_parts = []
                if known_faces_desc:
                    people_parts.extend(known_faces_desc)
                if unknown_person_count > 0:
                    if unknown_person_count == 1:
                        people_parts.append("شخص")
                    elif unknown_person_count == 2:
                        people_parts.append("شخصان")
                    elif unknown_person_count == 3:
                        people_parts.append("3 أشخاص")
                    else:
                        people_parts.append(f"{unknown_person_count} أشخاص")
                
                parts = []
                if people_parts:
                    parts.append(" و".join(people_parts))
                parts.extend(detected_objects)
                final = "رأيت: " + " و".join(parts) + "." if parts else ""
            else:
                # ══ Detailed mode: recognized faces + Gemini description, NO YOLO list ══
                final = ""
                if known_faces_desc:
                    final = "رأيت " + " و".join(known_faces_desc) + ". "
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                gemini  = get_gemini_response(pil_img, force=True)
                if gemini:
                    final += gemini
            
            speak_arabic(final or "لا أرى شيئاً مهماً.", priority=True)
        finally:
            is_processing = False

    elif warning_messages and safety_mode_enabled:
        # ✅ Safety alarm
        play_alarm_sound()
        time.sleep(0.5)
        speak_arabic("انتبه! " + " ".join(set(warning_messages)), priority=True)


# ============================================================
# Main
# ============================================================
def main():
    global process_request, running, yolo_model, is_processing
    global face_training_active, face_training_name, save_face_request, face_to_save_name

    # ✅ Raspberry Pi: Priority optimization
    if platform.system() == "Linux":
        try:
            os.nice(-10) # Give process higher priority (needs sudo)
        except:
            pass

    # ── Optional web panel ────────────────────────────────────
    web_process   = None
    ngrok_process = None
    ngrok_domain  = os.getenv("NGROK_DOMAIN", "")

    web_panel_path = os.path.join(BASE_DIR, "web_panel.py")
    if os.path.exists(web_panel_path):
        try:
            env = os.environ.copy()
            env["DISPLAY"] = ":0"
            env["PYTHONUNBUFFERED"] = "1"
            # ✅ LOG outputs to file to see why it fails
            with open(os.path.join(BASE_DIR, "web_log.txt"), "w") as log_f:
                web_process = subprocess.Popen(
                    [sys.executable, web_panel_path],
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=BASE_DIR  # Ensure correct working directory
                )
            print("🌐 Web Panel starting... checking log for status.")
            time.sleep(5) 
            
            if web_process.poll() is None:
                print("✅ Web panel is running in background.")
            else:
                print("⚠️ Web panel failed to start. Check web_log.txt")
                web_process = None

            ngrok_bin = subprocess.run(["which", "ngrok"],
                                        capture_output=True).stdout.strip()
            if ngrok_bin:
                if ngrok_domain:
                    ngrok_process = subprocess.Popen(
                        ["ngrok", "http", f"--url={ngrok_domain}", "5000"])
                    print(f"🌐 ngrok tunnel: https://{ngrok_domain}")
                else:
                    ngrok_process = subprocess.Popen(["ngrok", "http", "5000"])
                    print("🌐 ngrok started (check ngrok dashboard for URL)")
            else:
                print("ℹ️  ngrok not found — web panel on local network only (port 5000).")
        except Exception as e:
            print(f"Web panel error: {e}")

    try:
        # ── Load YOLO — CPU only ──────────────────────────────
        try:
            yolo_model = YOLO("yolov8n.pt")
            # ✅ Force all inference to CPU — prevents PyTorch from attempting
            # to load CUDA or ROCm libs which are not available on RPi and
            # would cause a segfault or ImportError at runtime.
            yolo_model.to("cpu")
            print("✅ YOLOv8n loaded on CPU")
        except Exception as e:
            print(f"YOLO load error: {e}")
            speak_arabic("خطأ في تحميل نموذج الكشف عن الأجسام.")
            return

        # ── Camera init ───────────────────────────────────────
        use_picamera = init_picamera()    # try picamera2 first
        cap = None
        if not use_picamera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                speak_arabic("خطأ في الكاميرا، تأكد من توصيلها.")
                return
            # ✅ Optimize OpenCV camera for RPi
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

        # ── Camera Display Init ───────────────────────────────
        # ✅ Ensure camera window is created and ready on top
        try:
            cv2.namedWindow("NORA - Smart Glasses", cv2.WINDOW_NORMAL)
            cv2.moveWindow("NORA - Smart Glasses", 0, 0)
            cv2.setWindowProperty("NORA - Smart Glasses", cv2.WND_PROP_TOPMOST, 1)
        except:
            pass

        # ✅ Pre-warm TTS cache in background before starting voice thread
        threading.Thread(target=_prewarm_common_phrases, daemon=True).start()
        # ✅ Detect audio player once at startup
        global _audio_player
        _audio_player = _detect_audio_player()
        threading.Thread(target=voice_trigger_thread, daemon=True).start()
        speak_arabic("مرحباً، نورا معك الآن، جاهزة لمساعدتك.")

        frame_count = 0
        while running:
            # ── Capture frame ─────────────────────────────────
            if use_picamera:
                frame = capture_frame()
                if frame is None:
                    print("picamera2: capture failed, retrying…")
                    time.sleep(1)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    print("cv2: camera disconnected, retrying…")
                    time.sleep(1)
                    cap = cv2.VideoCapture(0)
                    continue

            # ── Face training ──────────────────────────────────
            if save_face_request:
                if not face_training_active:
                    face_training_active      = True
                    face_training_name        = face_to_save_name
                    face_training_frame_count = 0
                    face_training_samples.clear()
                    face_training_ids.clear()
                frame = collect_face_frame(frame)

            # ── Live display on screen ─────────────────────────
            try:
                # ✅ Always bring window to front
                cv2.imshow("NORA - Smart Glasses", frame)
                cv2.setWindowProperty("NORA - Smart Glasses", cv2.WND_PROP_TOPMOST, 1)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                    break
            except Exception as display_err:
                # print(f"Display error: {display_err}")
                pass

            # ── Process special requests ───────────────────────
            with _state_lock:
                req = process_request
                if req:
                    process_request = False

            if req == "whatsapp_init":
                threading.Thread(
                    target=lambda: speak_arabic(init_whatsapp_session(), priority=True),
                    daemon=True).start()

            elif req == "sos":
                def run_sos():
                    speak_arabic("جاري إرسال رسالة استغاثة…", priority=True)
                    while is_speaking:
                        time.sleep(0.1)
                    result = send_sos_whatsapp_selenium()
                    speak_arabic(result, priority=True)
                threading.Thread(target=run_sos, daemon=True).start()

            elif req == "ocr" and not is_processing:
                is_processing = True
                def do_ocr(f):
                    global is_processing
                    try:
                        speak_arabic(read_text_from_frame(f), priority=True)
                    finally:
                        is_processing = False
                threading.Thread(target=do_ocr, args=(frame.copy(),), daemon=True).start()

            elif req == "currency" and not is_processing:
                is_processing = True
                def do_currency(f):
                    global is_processing
                    try:
                        speak_arabic(recognize_currency(f), priority=True)
                    finally:
                        is_processing = False
                threading.Thread(target=do_currency, args=(frame.copy(),), daemon=True).start()

            elif req and not is_processing:
                threading.Thread(target=process_frame_logic, args=(frame.copy(), True),
                                 daemon=True).start()

            # ── Background safety scan ─────────────────────────
            # ✅ _bg_scan_running prevents stacking YOLO calls when Pi is slow
            frame_count = (frame_count + 1) % 100000   # prevent int overflow
            check_interval = 30 if low_battery_mode else 20
            if (frame_count % check_interval == 0
                    and not is_processing
                    and not is_speaking
                    and not save_face_request
                    and not _bg_scan_running):
                def _bg_scan(f):
                    global _bg_scan_running
                    _bg_scan_running = True
                    try:
                        process_frame_logic(f, False)
                    finally:
                        _bg_scan_running = False
                threading.Thread(target=_bg_scan, args=(frame.copy(),), daemon=True).start()

            time.sleep(0.03)  # ~30 fps cap — keeps CPU free on Pi

        if cap:
            cap.release()
        if _picam2:
            _picam2.stop()

    except Exception:
        print("=" * 50)
        print("CRITICAL ERROR")
        print("=" * 50)
        traceback.print_exc()
        speak_arabic("حدث خطأ تقني أدى لتوقف البرنامج.")
    finally:
        if web_process:
            web_process.terminate()
        if ngrok_process:
            ngrok_process.terminate()


if __name__ == "__main__":
    main()
