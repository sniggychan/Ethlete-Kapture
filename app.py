import os
import uuid
import time
import tempfile
from datetime import datetime

from flask import Flask, request, jsonify, render_template_string, Response, send_file, stream_with_context
from flask_cors import CORS
import requests

# Optional CV libs
try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import mediapipe as mp
except Exception:
    mp = None

# Optional OpenAI python library (for true streaming when key present)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024  # 150MB uploads

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------
# Provider keys/endpoints
# -----------------------------
# Blackbox – FIRST
BLACKBOX_API_KEY = os.environ.get("BLACKBOX_API_KEY", "") or os.environ.get("BLACKBOX_KEY", "")
BLACKBOX_MODEL = os.environ.get("BLACKBOX_MODEL", "gpt-4o-mini")  # change if your tenant uses a different name
BLACKBOX_CHAT_URL = os.environ.get("BLACKBOX_CHAT_URL", "https://api.blackbox.ai/v1/chat/completions")

# Gemini
GEMINI_API_KEY = os.environ.get("AIzaSyB0D5G6JPyu9Ll_P_l7XVp3ABb-vpmUHeo", "") or \
                 os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"

# OpenAI (Python library)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

# AI/ML (OpenAI-compatible)
AI_ML_API_KEY = os.environ.get("AI_ML_API_KEY", "")
AI_ML_API_BASE = os.environ.get("AI_ML_API_BASE", "https://api.aimlapi.com/v1")
AI_ML_MODEL = os.environ.get("AI_ML_MODEL", "gpt-4o-mini")
AI_ML_CHAT_URL = f"{AI_ML_API_BASE.rstrip('/')}/chat/completions"

# -----------------------------
# In-memory state
# -----------------------------
SESSIONS = {}  # session_id -> [{"role":"user/assistant/system","content":"..."}]

USER = {
    "level": 1, "exp": 0, "streak": 0, "rating": "E",
    "stats": {"strength": 5, "agility": 5, "endurance": 5},
    "progress_history": [
        {"date": "2025-01-15", "max_height": 42, "reps": 28, "time": 32, "analysis": "Good form"},
        {"date": "2025-01-16", "max_height": 45, "reps": 30, "time": 30, "analysis": "Improved"},
        {"date": "2025-01-17", "max_height": 43, "reps": 29, "time": 31, "analysis": "Consistent"},
        {"date": "2025-01-18", "max_height": 47, "reps": 32, "time": 29, "analysis": "Excellent"},
        {"date": "2025-01-19", "max_height": 46, "reps": 31, "time": 30, "analysis": "Strong"},
    ],
    "daily_tasks": ["Record a vertical jump", "Complete 20 sit-ups", "Run a shuttle", "Track endurance"],
}

# -----------------------------
# Helpers
# -----------------------------
def ek_system_prompt(lang="en"):
    base = (
        "You are Ethlete Kapture AI Coach. Be clear, specific, and actionable. "
        "Prefer bullet points, sets/reps, rest, RPE, and form cues. "
        "If user mentions injury/pain, add safety guidance and recommend consulting a professional. "
        "Be concise unless asked for detail."
    )
    if lang.lower() == "hi":
        base += " उत्तर हिंदी में दें।"
    if lang.lower() == "ta":
        base += " பதிலை தமிழில் அளிக்கவும்."
    return {"role": "system", "content": base}

def get_session(sid: str, lang="en"):
    if not sid or sid not in SESSIONS:
        sid = sid or uuid.uuid4().hex
        SESSIONS[sid] = [ek_system_prompt(lang)]
    return sid, SESSIONS[sid]

def clamp_stat(v: int) -> int:
    return max(0, min(v, 100))

def rating_for_level(level: int) -> str:
    return ['E', 'D', 'C', 'B', 'A', 'S'][min(max((level - 1) // 5, 0), 5)]

def level_up(user: dict, exp_gain: int):
    exp_total = user["exp"] + exp_gain
    while exp_total >= 100 * user["level"]:
        exp_total -= 100 * user["level"]
        user["level"] += 1
    user["exp"] = exp_total
    user["rating"] = rating_for_level(user["level"])

# -----------------------------
# Provider callers
# -----------------------------
def call_blackbox(messages_or_prompt, temperature=0.7, max_tokens=600):
    """Blackbox chat (non-stream); returns full text."""
    if not BLACKBOX_API_KEY:
        return False, "Blackbox key missing"
    headers = {"Authorization": f"Bearer {BLACKBOX_API_KEY}", "Content-Type": "application/json"}

    # If we got a list of messages, send as chat; else treat as prompt
    if isinstance(messages_or_prompt, list):
        payload = {
            "model": BLACKBOX_MODEL,
            "messages": messages_or_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            r = requests.post(BLACKBOX_CHAT_URL, headers=headers, json=payload, timeout=30)
            if r.ok:
                data = r.json()
                txt = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if txt: return True, txt
        except Exception as e:
            return False, f"Blackbox error: {e}"
    else:
        prompt = messages_or_prompt
        try:
            # completion schema fallback
            url = "https://api.blackbox.ai/v1/completions"
            payload_alt = {"model": BLACKBOX_MODEL, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
            r2 = requests.post(url, headers=headers, json=payload_alt, timeout=30)
            if r2.ok:
                data = r2.json()
                txt = data.get("choices", [{}])[0].get("text", "").strip()
                if txt: return True, txt
        except Exception as e:
            return False, f"Blackbox error: {e}"
    return False, "Blackbox failed"

def call_gemini(prompt: str, temperature=0.7, max_tokens=600):
    if not GEMINI_API_KEY:
        return False, "Gemini key missing"
    try:
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        r = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, timeout=30)
        if not r.ok: return False, f"Gemini HTTP {r.status_code}"
        data = r.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        txt = "".join(p.get("text", "") for p in parts).strip()
        return (True, txt) if txt else (False, "Gemini empty")
    except Exception as e:
        return False, f"Gemini error: {e}"

def call_openai_stream(messages, temperature=0.7, max_tokens=600):
    """OpenAI Python library streaming; yields tokens and returns final text."""
    if not OPENAI_CLIENT:
        return None
    try:
        stream = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        def generator():
            final = []
            for event in stream:
                delta = event.choices[0].delta
                if delta and getattr(delta, "content", None):
                    token = delta.content
                    final.append(token)
                    yield token
            return "".join(final)
        return generator()
    except Exception:
        return None

def call_aiml(prompt: str, temperature=0.7, max_tokens=600):
    if not AI_ML_API_KEY:
        return False, "AI/ML key missing"
    payload = {"model": AI_ML_MODEL, "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "max_tokens": max_tokens}
    try:
        h1 = {"Authorization": f"Bearer {AI_ML_API_KEY}", "Content-Type": "application/json"}
        r1 = requests.post(AI_ML_CHAT_URL, headers=h1, json=payload, timeout=30)
        if r1.ok:
            data = r1.json()
            txt = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if txt: return True, txt
        h2 = {"x-api-key": AI_ML_API_KEY, "Content-Type": "application/json"}
        r2 = requests.post(AI_ML_CHAT_URL, headers=h2, json=payload, timeout=30)
        if r2.ok:
            data = r2.json()
            txt = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if txt: return True, txt
        return False, "AI/ML HTTP error"
    except Exception as e:
        return False, f"AI/ML error: {e}"

def coach_fallback(messages):
    q = messages[-1]["content"].lower()
    tips = []
    if "diet" in q or "nutrition" in q: tips.append("Base meals on protein + veggies + whole carbs. 2–3L water/day.")
    if "jump" in q: tips.append("Add CMJ, depth drops, box jumps; 2–3x/week. Focus on fast SSC.")
    if "strength" in q: tips.append("3–5x5 squat/hinge/press at RPE 7–8, tight form.")
    if not tips: tips.append("Form, progressive overload, sleep, and recovery are key.")
    return " ".join(tips)

def messages_to_prompt(messages):
    out = []
    for m in messages[-16:]:
        role = m["role"].capitalize()
        out.append(f"{role}: {m['content']}")
    out.append("Assistant:")
    return "\n".join(out)

def ai_stream(messages, temperature=0.7, max_tokens=600):
    """
    Streaming generator: Blackbox (chunked) → Gemini (chunked) → OpenAI (true stream) → AI/ML (chunked) → local.
    Yields text chunks; caller is responsible for saving final text to history.
    """
    # 1) Blackbox first (non-stream; chunk it)
    ok, txt = call_blackbox(messages, temperature, max_tokens)
    if ok and txt:
        for i in range(0, len(txt), 40):
            yield txt[i:i+40]
        return

    # 2) Gemini (non-stream; chunk)
    prompt = messages_to_prompt(messages)
    ok, txt = call_gemini(prompt, temperature, max_tokens)
    if ok and txt:
        for i in range(0, len(txt), 40):
            yield txt[i:i+40]
        return

    # 3) OpenAI true streaming (if available)
    if OPENAI_CLIENT:
        gen = call_openai_stream(messages, temperature, max_tokens)
        if gen:
            for token in gen:
                yield token
            return

    # 4) AI/ML (non-stream; chunk)
    ok, txt = call_aiml(prompt, temperature, max_tokens)
    if ok and txt:
        for i in range(0, len(txt), 40):
            yield txt[i:i+40]
        return

    # 5) Local coach fallback
    txt = coach_fallback(messages)
    for i in range(0, len(txt), 40):
        yield txt[i:i+40]

# -----------------------------
# Vision utilities
# -----------------------------
def save_overlay_image(img_bgr):
    if cv2 is None:
        return None
    rid = f"{uuid.uuid4().hex}.png"
    path = os.path.join(RESULT_DIR, rid)
    try:
        cv2.imwrite(path, img_bgr)
        return f"/result/{rid}"
    except Exception:
        return None

def draw_pose(image_bgr, res_pose=None):
    if mp is None or cv2 is None:
        return image_bgr
    try:
        du = mp.solutions.drawing_utils
        ds = mp.solutions.drawing_styles
        du.draw_landmarks(
            image_bgr, res_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=ds.get_default_pose_landmarks_style()
        )
    except Exception:
        pass
    return image_bgr

def angle_deg(a, b, c):
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb)) + 1e-6
    cosang = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

# -----------------------------
# Photo analysis
# -----------------------------
def analyze_image_bytes(img_bytes: bytes):
    if cv2 is None or np is None:
        return {
            "success": True,
            "analysis": "Image received. Install opencv-python and numpy for pose angles.",
            "metrics": {"max_height": 0.0, "reps": 1, "time": 0.0},
            "segments": [], "is_cheat": False, "anomalies": [],
            "feedback": "Static image—no jump metrics.",
            "overlay_url": None
        }
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    overlay_url = None
    analysis = "Image analyzed."

    if mp is not None:
        pose = mp.solutions.pose.Pose(static_image_mode=True)
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            LHIP = mp.solutions.pose.PoseLandmark.LEFT_HIP.value
            LKNE = mp.solutions.pose.PoseLandmark.LEFT_KNEE.value
            LANK = mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value
            RHIP = mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
            RKNE = mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value
            RANK = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value
            LSHO = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
            RSHO = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
            def XY(i): return (lm[i].x * w, lm[i].y * h)
            kneeL = angle_deg(XY(LHIP), XY(LKNE), XY(LANK))
            kneeR = angle_deg(XY(RHIP), XY(RKNE), XY(RANK))
            hipL  = angle_deg(XY(LSHO), XY(LHIP), XY(LKNE))
            hipR  = angle_deg(XY(RSHO), XY(RHIP), XY(RKNE))
            knee = round((kneeL + kneeR) / 2.0, 1)
            hip = round((hipL + hipR) / 2.0, 1)
            cues = []
            cues.append("Knee flexion adequate" if knee < 150 else "Bend knees more (~130–150°)")
            cues.append("Torso neutral" if 160 <= hip <= 180 else "Keep torso 160–180°")
            analysis = f"Detected person. Avg knee {knee}°, hip {hip}°. " + "; ".join(cues) + "."
            img_ov = draw_pose(img.copy(), res_pose=res)
            cv2.putText(img_ov, f"Knee {knee}°, Hip {hip}°", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            overlay_url = save_overlay_image(img_ov)

    return {
        "success": True,
        "analysis": analysis,
        "metrics": {"max_height": 0.0, "reps": 1, "time": 0.0},
        "segments": [], "is_cheat": False, "anomalies": [],
        "feedback": "Static image analyzed for posture.",
        "overlay_url": overlay_url
    }

# -----------------------------
# Video analysis
# -----------------------------
def analyze_video_file(path: str):
    if cv2 is None or np is None:
        return {
            "success": True,
            "analysis": "Video received. Install opencv-python and numpy for analysis.",
            "metrics": {"max_height": 45.0, "reps": 1, "time": 10.0},
            "segments": ["Rep 1 at t=2.5s (45.0cm)"],
            "is_cheat": False, "anomalies": [],
            "feedback": "Default metrics used (OpenCV not installed).",
            "overlay_url": None
        }
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"success": False, "error": "Could not open video"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if frames else 0.0

    use_pose = mp is not None
    hip_series = []
    frame_indices = []
    step = 2

    if use_pose:
        pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    else:
        fgbg = cv2.createBackgroundSubtractorMOG2()

    idx = 0
    frame_h = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % step != 0:
            idx += 1; continue
        frame_h, frame_w = frame.shape[:2]
        if use_pose:
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                LHIP = mp.solutions.pose.PoseLandmark.LEFT_HIP.value
                RHIP = mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
                hip_y = (lm[LHIP].y + lm[RHIP].y) / 2.0
                hip_series.append(hip_y)
                frame_indices.append(idx)
        else:
            fgmask = fgbg.apply(frame)
            ys, xs = np.where(fgmask > 127)
            if len(ys) > 50:
                hip_series.append(np.median(ys) / frame_h)
                frame_indices.append(idx)
        idx += 1
    cap.release()

    if len(hip_series) < 8:
        overlay_url = None
        cap2 = cv2.VideoCapture(path)
        ok, frame = cap2.read()
        if ok: overlay_url = save_overlay_image(frame)
        cap2.release()
        return {
            "success": True,
            "analysis": "Insufficient pose/motion; returning conservative defaults.",
            "metrics": {"max_height": 42.0, "reps": 1, "time": round(duration, 1)},
            "segments": ["Rep 1 at t=2.50s (42.0cm)"],
            "is_cheat": False, "anomalies": ["Low detection confidence"],
            "feedback": f"Feedback: Jump 42.0cm above avg 40.0cm, Reps 1 vs 1, Time {round(duration,1)}s vs 30.0s.",
            "overlay_url": overlay_url
        }

    hip_series = np.array(hip_series)
    hip_s = np.convolve(hip_series, np.ones(5)/5.0, mode='same')
    baseline = float(np.quantile(hip_s, 0.85))
    thr = 0.03
    airborne = hip_s < (baseline - thr)

    segments_idx = []
    in_seg = False
    start = 0
    for i, val in enumerate(airborne):
        if val and not in_seg: in_seg = True; start = i
        if in_seg and (not val or i == len(airborne)-1):
            end = i
            if end - start >= max(3, int((fps/step) * 0.18)):
                segments_idx.append((start, end))
            in_seg = False

    reps = len(segments_idx) if segments_idx else 0
    seg_strings = []
    max_height_cm = 0.0
    overlay_url = None

    if segments_idx:
        first_s, first_e = segments_idx[0]
        mid_idx = first_s + (first_e - first_s) // 2
        frame_number = frame_indices[mid_idx] if mid_idx < len(frame_indices) else frame_indices[first_s]
        cap3 = cv2.VideoCapture(path)
        cap3.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, mid_frame = cap3.read()
        cap3.release()
        if ok:
            if use_pose:
                pose2 = mp.solutions.pose.Pose(static_image_mode=True)
                res2 = pose2.process(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB))
                if res2.pose_landmarks:
                    mid_frame = draw_pose(mid_frame, res_pose=res2)
            overlay_url = save_overlay_image(mid_frame)

    for j, (s, e) in enumerate(segments_idx[:6], 1):
        T = (e - s) * (step / fps)
        h_cm = 122.6 * (T ** 2)  # h = g*T^2/8, in cm
        max_height_cm = max(max_height_cm, h_cm)
        t_mark = round(frame_indices[s] / fps, 2) if s < len(frame_indices) else round(s * (step / fps), 2)
        seg_strings.append(f"Rep {j} at t={t_mark:.2f}s ({h_cm:.1f}cm)")

    if reps == 0:
        amp = max(0.0, float(baseline - float(np.min(hip_s))))
        h_cm = amp * 200.0
        max_height_cm = max(max_height_cm, h_cm)
        reps = 1
        seg_strings.append(f"Rep 1 at t=2.50s ({h_cm:.1f}cm)")

    anomalies = []
    if max_height_cm > 80: anomalies.append("Unusually high jump")
    if reps > 20: anomalies.append("Unusually high rep count")
    is_cheat = len(anomalies) > 0

    analysis = f"Estimated {reps} jump(s). Max jump ≈ {max_height_cm:.1f} cm. Flight-time–based estimate."
    feedback = (
        f"Feedback: Jump {max_height_cm:.1f}cm "
        f"{'above' if max_height_cm > 40 else 'below'} avg 40.0cm, "
        f"Reps {reps} vs 25, Time {round(duration,1)}s vs 30.0s."
    )
    return {
        "success": True,
        "analysis": analysis,
        "metrics": {"max_height": round(max_height_cm, 1), "reps": int(reps), "time": round(duration, 1)},
        "segments": seg_strings,
        "is_cheat": is_cheat, "anomalies": anomalies,
        "feedback": feedback,
        "overlay_url": overlay_url
    }

def analyze_media(file_storage):
    name = (file_storage.filename or "").lower()
    content = file_storage.read()
    file_storage.seek(0)
    if any(name.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
        return analyze_image_bytes(content)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1] or ".mp4") as tmp:
        tmp.write(content); tmp.flush()
        path = tmp.name
    try:
        return analyze_video_file(path)
    finally:
        try: os.remove(path)
        except Exception: pass

# -----------------------------
# Routes
# -----------------------------
@app.route("/static/<path:filename>", endpoint="static")
def serve_static(filename):
    if filename == "config.js":
        js = f'const CONFIG = {{ API_BASE: "{request.host_url.rstrip("/")}" }};'
        return Response(js, mimetype="application/javascript")
    return Response("", status=404)

@app.route("/")
@app.route("/intex.html")
def serve_intex():
    try:
        with open("intex.html", "r", encoding="utf-8") as f:
            html = f.read()
        return render_template_string(html)
    except Exception as e:
        return f"Could not load intex.html: {e}", 500

@app.route("/result/<rid>")
def serve_result(rid):
    path = os.path.join(RESULT_DIR, rid)
    if not os.path.isfile(path):
        return "Not found", 404
    return send_file(path, mimetype="image/png", conditional=True)

@app.route("/dashboard", methods=["GET"])
def dashboard():
    return jsonify(USER)

@app.route("/upload", methods=["POST"])
def upload():
    age_group = (request.form.get("age_group") or "").strip()
    gender = (request.form.get("gender") or "").strip()
    file = request.files.get("file")
    if not age_group or not gender or not file:
        return jsonify({"success": False, "error": "Missing age_group, gender or file"}), 400

    result = analyze_media(file)
    if not result.get("success"):
        return jsonify({"success": False, "error": result.get("error", "Analysis failed")}), 500

    max_h = float(result["metrics"]["max_height"])
    reps = int(result["metrics"]["reps"])
    time_s = float(result["metrics"]["time"])

    # Coaching feedback: compact, actionable
    prompt = (
        f"You are a sports coach. Athlete ({gender}, {age_group}) vertical-jump test:\n"
        f"- Jump height: {max_h:.1f} cm\n- Reps: {reps}\n- Time: {time_s:.1f} s\n\n"
        "Return: 1) 2–3 line performance summary, 2) three specific form fixes, 3) a 1-week micro-plan."
    )
    messages = [ek_system_prompt("en"), {"role":"user","content":prompt}]
    # Quick non-stream generation for upload
    full_text = []
    for chunk in ai_stream(messages, temperature=0.6, max_tokens=600):
        full_text.append(chunk)
    txt = "".join(full_text)
    result["analysis"] = (result.get("analysis") or "") + ("\n" + txt if txt else "")
    result["model_used"] = "multi"

    # Update user
    exp_gain = int(max_h + reps + (time_s or 0))
    level_up(USER, exp_gain)
    USER["streak"] = USER.get("streak", 0) + 1
    USER["stats"]["strength"] = clamp_stat(USER["stats"]["strength"] + max(1, reps // 10))
    USER["stats"]["agility"] = clamp_stat(USER["stats"]["agility"] + max(1, int(max_h // 10)))
    USER["stats"]["endurance"] = clamp_stat(USER["stats"]["endurance"] + max(1, int((time_s or 0) // 10)))
    USER["progress_history"] = [{
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "max_height": max_h, "reps": reps, "time": time_s,
        "analysis": (result["analysis"] or "")[:100]
    }] + USER["progress_history"]
    USER["progress_history"] = USER["progress_history"][:10]
    result["user"] = USER
    return jsonify(result)

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    # Streaming response (plain text chunks)
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    lang = (data.get("lang") or "en").strip().lower()
    sid = (data.get("sid") or "").strip()
    if not msg:
        return jsonify({"error": "message is required"}), 400

    sid, history = get_session(sid, lang)
    # Ensure language system prompt
    if history and history[0]["role"] == "system":
        history[0] = ek_system_prompt(lang)

    history.append({"role": "user", "content": msg})

    def gen():
        yield f"[SESSION]{sid}\n"
        acc = []
        for chunk in ai_stream(history, temperature=0.7, max_tokens=800):
            acc.append(chunk)
            yield chunk
        # Save assistant message to memory
        history.append({"role": "assistant", "content": "".join(acc)})

    headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return Response(stream_with_context(gen()), headers=headers)

@app.route("/chat/clear", methods=["POST"])
def chat_clear():
    data = request.get_json(silent=True) or {}
    sid = (data.get("sid") or "").strip()
    lang = (data.get("lang") or "en").strip().lower()
    if sid and sid in SESSIONS:
        del SESSIONS[sid]
    new_sid, _ = get_session("", lang)
    return jsonify({"sid": new_sid})

@app.route("/ai_advisor", methods=["POST"])
def ai_advisor():
    data = request.get_json(silent=True) or {}
    q = (data.get("query") or "").strip()
    if not q: return jsonify({"error": "query is required"}), 400
    messages = [ek_system_prompt("en"), {"role":"user","content": f"Fitness micro-advice (<=150 chars): {q}"}]
    acc = []
    for chunk in ai_stream(messages, temperature=0.7, max_tokens=140):
        acc.append(chunk)
    return jsonify({"response": ("".join(acc) or '')[:200], "model_used": "multi"})

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Open http://127.0.0.1:{port}/intex.html")
    app.run(host="0.0.0.0", port=port, debug=True)