import os
import functools
import logging
from datetime import timedelta
from flask import Flask, request, render_template_string, session, redirect, url_for, flash
from werkzeug.security import check_password_hash, generate_password_hash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))
# مفتاح سري لتأمين الجلسات (Sessions)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(minutes=10) # تنتهي الجلسة بعد 10 دقائق

# إعداد نظام التسجيل (Logging)
logging.basicConfig(
    filename='nora.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# إعداد الـ Rate Limiter لمنع محاولات التخمين
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "10 per hour"],
    storage_uri="memory://"
)

ENV_FILE = '.env'

def read_env():
    env_vars = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    key, val = line.strip().split('=', 1)
                    env_vars[key] = val
    return env_vars

def write_env(env_vars):
    with open(ENV_FILE, 'w', encoding='utf-8') as f:
        for k, v in env_vars.items():
            f.write(f"{k}={v}\n")

# ✅ ديكوراتور لحماية الصفحات (تطلب تسجيل الدخول)
def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==========================================
# 🎨 القوالب (Templates)
# ==========================================

# ستايل مشترك (الزجاج الشفاف والخلفية)
COMMON_STYLE = """
<style>
    body { 
        font-family: Tahoma, Arial, sans-serif; 
        margin: 0; 
        padding: 20px; 
        text-align: center; 
        background: url('/static/bg.jpg') no-repeat center center fixed, #111;
        background-size: cover; 
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
    }
    .container { 
        background: rgba(15, 23, 42, 0.8); 
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px; 
        border-radius: 20px; 
        box-shadow: 0 10px 40px rgba(0,0,0,0.5); 
        max-width: 450px; 
        width: 100%;
    }
    h1 { color: #f8fafc; font-size: 28px; margin-bottom: 25px; }
    label { display: block; margin-top: 15px; text-align: right; font-weight: bold; color: #cbd5e1; }
    input { 
        width: 100%; 
        padding: 12px; 
        margin-top: 8px; 
        background: rgba(255, 255, 255, 0.05); 
        border: 1px solid rgba(255,255,255,0.2); 
        border-radius: 10px; 
        color: white; 
        outline: none; 
        font-size: 15px;
    }
    input:focus { border-color: #3b82f6; background: rgba(255,255,255,0.1); }
    .btn { 
        margin-top: 30px; 
        width: 100%; 
        padding: 14px; 
        background: linear-gradient(135deg, #3b82f6, #2563eb); 
        color: white; 
        border: none; 
        border-radius: 10px; 
        font-size: 18px; 
        font-weight: bold; 
        cursor: pointer; 
        transition: 0.3s; 
    }
    .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(59,130,246,0.4); }
    .btn-logout { background: #ef4444; border-radius: 8px; padding: 5px 15px; font-size: 14px; margin-top: 20px; width: auto; display: inline-block; text-decoration: none; color: white; }
    .error { color: #f87171; margin-bottom: 20px; font-weight: bold; }
    .success { color: #10b981; margin-bottom: 20px; font-weight: bold; }
</style>
"""

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>تسجيل الدخول - نظارة نورا 👓</title>
    """ + COMMON_STYLE + """
</head>
<body>
    <div class="container">
        <h1>تسجيل الدخول 👓</h1>
        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}
        <form method="POST">
            <label>اسم المستخدم:</label>
            <input type="text" name="email" required placeholder="مثال: NORA">
            
            <label>كلمة المرور:</label>
            <input type="password" name="password" required placeholder="******">
            
            <button type="submit" class="btn">دخول للنظام 🚀</button>
        </form>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>لوحة تحكم نورا 👓</title>
    """ + COMMON_STYLE + """
</head>
<body>
    <div class="container">
        <div style="text-align: left;"><a href="/logout" class="btn-logout">تسجيل خروج ⬅️</a></div>
        <h1>إعدادات نورا الذكية 👓</h1>
        {% if success %}
        <p class="success">تم حفظ الإعدادات بنجاح! ✅</p>
        {% endif %}
        <form method="POST">
            <label>مفتاح جوجل (Gemini API Key):</label>
            <input type="password" name="GEMINI_API_KEY" value="{{ env_vars.get('GEMINI_API_KEY', '') }}" placeholder="سري للغاية">
            
            <label>رقم هاتف الطوارئ (واتساب):</label>
            <input type="text" name="EMERGENCY_PHONE" value="{{ env_vars.get('EMERGENCY_PHONE', '') }}">
            
            <label>بريد الطوارئ (المستقبل):</label>
            <input type="email" name="EMERGENCY_EMAIL" value="{{ env_vars.get('EMERGENCY_EMAIL', '') }}">
            
            <label>بريد النظارة (المرسل):</label>
            <input type="email" name="SENDER_EMAIL" value="{{ env_vars.get('SENDER_EMAIL', '') }}">

            <label>باسورد إيميل النظارة:</label>
            <input type="password" name="SENDER_APP_PASSWORD" value="{{ env_vars.get('SENDER_APP_PASSWORD', '') }}" placeholder="سري للغاية">

            <button type="submit" class="btn">حفظ الإعدادات 💾</button>
        </form>

        <hr style="border-color: rgba(255,255,255,0.1); margin: 35px 0;">
        <h2 style="font-size: 20px; color: #f8fafc;">تغيير بيانات الدخول 🔐</h2>
        {% if cred_success %}
        <p class="success">تم تغيير بيانات الدخول بنجاح! ✅</p>
        {% endif %}
        {% if cred_error %}
        <p class="error">{{ cred_error }}</p>
        {% endif %}
        <form method="POST" action="/change-credentials">
            <label>اسم المستخدم الجديد:</label>
            <input type="text" name="new_username" placeholder="مثال: NORA">

            <label>كلمة المرور الجديدة:</label>
            <input type="password" name="new_password" placeholder="******">

            <button type="submit" class="btn" style="background: linear-gradient(135deg, #10b981, #059669);">حفظ بيانات الدخول 🔐</button>
        </form>
    </div>
</body>
</html>
"""

# ==========================================
# 🛤️ المسارات (Routes)
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    error = None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        env_vars = read_env()
        valid_email = env_vars.get('LOGIN_EMAIL', 'norasamrtglasses@gmail.com')
        valid_password_hash = env_vars.get('LOGIN_PASSWORD_HASH', '')
        
        if email == valid_email and check_password_hash(valid_password_hash, password):
            session.permanent = True
            session['logged_in'] = True
            logging.info(f"Successful login for user: {email}")
            return redirect(url_for('index'))
        else:
            error = "بيانات الدخول غير صحيحة! يرجى المحاولة مرة أخرى."
            logging.warning(f"Failed login attempt for user: {email} from {request.remote_addr}")
            
    return render_template_string(LOGIN_TEMPLATE, error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    success = False
    env_vars = read_env()
    
    if request.method == 'POST':
        env_vars['GEMINI_API_KEY'] = request.form.get('GEMINI_API_KEY', '')
        env_vars['EMERGENCY_PHONE'] = request.form.get('EMERGENCY_PHONE', '')
        env_vars['EMERGENCY_EMAIL'] = request.form.get('EMERGENCY_EMAIL', '')
        env_vars['SENDER_EMAIL'] = request.form.get('SENDER_EMAIL', '')
        env_vars['SENDER_APP_PASSWORD'] = request.form.get('SENDER_APP_PASSWORD', '')
        
        write_env(env_vars)
        success = True

    return render_template_string(DASHBOARD_TEMPLATE, env_vars=env_vars, success=success,
                                   cred_success=False, cred_error=None)

@app.route('/change-credentials', methods=['POST'])
@login_required
def change_credentials():
    new_username = request.form.get('new_username', '').strip()
    new_password = request.form.get('new_password', '')

    env_vars = read_env()
    cred_error = None
    cred_success = False

    if not new_username and not new_password:
        cred_error = "يرجى إدخال اسم المستخدم أو كلمة المرور الجديدة."
    else:
        if new_username:
            env_vars['LOGIN_EMAIL'] = new_username
            logging.info(f"Login username changed to: {new_username}")
        if new_password:
            env_vars['LOGIN_PASSWORD_HASH'] = generate_password_hash(new_password)
            logging.info("Login password hash updated.")
        write_env(env_vars)
        cred_success = True

    return render_template_string(DASHBOARD_TEMPLATE, env_vars=env_vars,
                                   success=False, cred_success=cred_success, cred_error=cred_error)


if __name__ == '__main__':
    print("=========================================")
    print(" Nora secure server is running...")
    print(" Default Login Data:")
    print(" Email: norasamrtglasses@gmail.com")
    print(" Password: AUT-SMART GLASSES")
    print("=========================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
