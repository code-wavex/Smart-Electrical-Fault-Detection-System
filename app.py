import os, uuid, joblib, numpy as np
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature
import simulate

app = Flask(__name__); CORS(app)
ser = URLSafeTimedSerializer(os.getenv("SECRET_KEY", "dev-secret-change-me"))
NODE_KEY = os.getenv("NODE_KEY", "node-secret-123")
now = lambda: (datetime.utcnow() + timedelta(hours=2)).strftime("%H:%M:%S")  # CAT (UTC+2)

# ---------- Storage: MongoDB Atlas if MONGO_URI set, otherwise in-memory ----------
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    from pymongo import MongoClient
    db = MongoClient(MONGO_URI)["faultdb"]
    def save(col, doc): db[col].insert_one(dict(doc))
    def fetch(col, flt=None): return list(db[col].find(flt or {}, {"_id": 0}).sort("ts", -1).limit(30))
    def update(col, id_, fields): db[col].update_one({"id": id_}, {"$set": fields})
else:
    store = {"readings": [], "alerts": [], "reports": []}
    def save(col, doc): store[col].insert(0, dict(doc))
    def fetch(col, flt=None):
        return [d for d in store[col] if all(d.get(k) == v for k, v in (flt or {}).items())][:30]
    def update(col, id_, fields):
        for d in store[col]:
            if d.get("id") == id_: d.update(fields)

# ---------- Users and auth ----------
ROLES = ["administrator", "safety_officer", "resident"]
USERS = {
    "admin":    {"pw": generate_password_hash("admin123"),    "role": "administrator"},
    "officer":  {"pw": generate_password_hash("officer123"),  "role": "safety_officer"},
    "resident": {"pw": generate_password_hash("resident123"), "role": "resident"},
}

def require(*roles):
    def deco(f):
        @wraps(f)
        def wrapper(*a, **kw):
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            try: data = ser.loads(token, max_age=8 * 3600)
            except BadSignature: return jsonify(error="Please log in again"), 401
            if roles and data["role"] not in roles: return jsonify(error="Not allowed for your role"), 403
            request.user = data
            return f(*a, **kw)
        return wrapper
    return deco

@app.post("/api/login")
def api_login():
    d = request.json or {}
    u = USERS.get(d.get("username", "").strip().lower())
    if not u or not check_password_hash(u["pw"], d.get("password", "")):
        return jsonify(error="Invalid username or password"), 401
    uname = d["username"].strip().lower()
    return jsonify(token=ser.dumps({"username": uname, "role": u["role"]}), username=uname, role=u["role"])

@app.post("/api/register")  # residents self-register
def api_register():
    d = request.json or {}
    uname, pw = d.get("username", "").strip().lower(), d.get("password", "")
    if len(uname) < 3 or len(pw) < 6: return jsonify(error="Username 3+ chars, password 6+ chars"), 400
    if uname in USERS: return jsonify(error="Username already taken"), 409
    USERS[uname] = {"pw": generate_password_hash(pw), "role": "resident"}
    return jsonify(ok=True), 201

@app.get("/api/users")
@require("administrator")
def api_users(): return jsonify([{"username": k, "role": v["role"]} for k, v in USERS.items()])

@app.post("/api/users")
@require("administrator")
def api_create_user():
    d = request.json or {}
    uname, pw, role = d.get("username", "").strip().lower(), d.get("password", ""), d.get("role")
    if len(uname) < 3 or len(pw) < 6 or role not in ROLES: return jsonify(error="Invalid details"), 400
    if uname in USERS: return jsonify(error="Username already taken"), 409
    USERS[uname] = {"pw": generate_password_hash(pw), "role": role}
    return jsonify(ok=True), 201

# ---------- Model ----------
le = joblib.load("labels.joblib"); winner = joblib.load("winner.joblib")
if winner == "rf":
    model = joblib.load("model_rf.joblib")
    predict = lambda x: model.predict(x)[0]
else:
    import tensorflow as tf
    model = tf.keras.models.load_model("model_nn.keras"); sc = joblib.load("scaler.joblib")
    predict = lambda x: model.predict(sc.transform(x), verbose=0).argmax(1)[0]

def send_sms(number, text):
    print(f"[SMS -> {number}] {text}")   # replace with Airtel/MTN gateway API

def classify_and_store(node_id, v, i, leak, pf):
    label = le.inverse_transform([predict(np.array([[v, i, leak, pf]]))])[0]
    doc = dict(node_id=node_id, voltage=round(v, 1), current=round(i, 2), leakage_ma=round(leak, 1),
               power_factor=round(pf, 2), fault=label, time=now(), ts=datetime.utcnow().timestamp())
    save("readings", doc)
    if label != "normal":
        save("alerts", {**doc, "id": uuid.uuid4().hex[:8], "status": "open"})
        try: send_sms("+260XXXXXXXXX", f"ALERT {label.upper()} at {node_id}")
        except Exception as e: print("SMS failed:", e)   # alert is still stored
    return doc

# ---------- Data endpoints ----------
@app.post("/api/predict")  # sensor nodes authenticate with an API key
def api_predict():
    if request.headers.get("X-Node-Key") != NODE_KEY: return jsonify(error="Invalid node key"), 401
    d = request.get_json(silent=True) or {}
    try:
        vals = [float(d[k]) for k in ("voltage", "current", "leakage_ma", "power_factor")]
        node_id = str(d["node_id"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="node_id, voltage, current, leakage_ma and power_factor are required"), 400
    return jsonify(classify_and_store(node_id, *vals))

@app.get("/api/simulate/<node_id>")
@require("administrator", "safety_officer")
def api_simulate(node_id):
    label = np.random.choice(simulate.CLASSES, p=[0.6, 0.1, 0.1, 0.1, 0.1])
    return jsonify(classify_and_store(node_id, *simulate.sample(label)))

@app.get("/api/readings")
@require("administrator", "safety_officer")
def api_readings(): return jsonify(fetch("readings"))

@app.get("/api/alerts")
@require("administrator", "safety_officer")
def api_alerts(): return jsonify(fetch("alerts"))

@app.post("/api/alerts/<alert_id>/ack")
@require("administrator", "safety_officer")
def api_ack(alert_id):
    update("alerts", alert_id, {"status": "acknowledged", "by": request.user["username"]})
    return jsonify(ok=True)

@app.post("/api/reports")
@require("administrator", "safety_officer", "resident")
def api_reports():
    d = request.get_json(silent=True) or {}
    loc, desc = str(d.get("location", "")).strip(), str(d.get("description", "")).strip()
    if not loc or not desc: return jsonify(error="Location and description are required"), 400
    if len(loc) > 100 or len(desc) > 500: return jsonify(error="Location max 100, description max 500 characters"), 400
    doc = dict(id=uuid.uuid4().hex[:8], username=request.user["username"], location=loc,
               type=str(d.get("type", "")), description=desc, status="submitted",
               time=now(), ts=datetime.utcnow().timestamp())
    save("reports", doc)
    return jsonify(ok=True), 201

@app.get("/api/reports")
@require("administrator", "safety_officer", "resident")
def api_get_reports():
    if request.user["role"] == "resident":
        return jsonify(fetch("reports", {"username": request.user["username"]}))
    return jsonify(fetch("reports"))

# ---------- Dashboard ----------
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Fault Detection System</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#1b2430}
header{background:#12355b;color:#fff;padding:12px 22px;display:flex;justify-content:space-between;align-items:center}
header button{width:auto;background:#fff;color:#12355b;padding:6px 12px}
main{display:grid;grid-template-columns:2fr 1fr;gap:16px;padding:16px}
.card{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px #0002;margin-bottom:16px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:6px 8px;text-align:left;border-bottom:1px solid #e3e7eb}
.normal{color:#1a7f37}.fault{color:#c62828;font-weight:600}
input,select,textarea,button{width:100%;margin:4px 0;padding:8px;box-sizing:border-box}
button{background:#12355b;color:#fff;border:0;border-radius:4px;cursor:pointer}
#login{max-width:340px;margin:60px auto}.err{color:#c62828;font-size:13px}.hint{font-size:12px;color:#667}
.sm{width:auto;padding:3px 8px;font-size:12px}
</style></head><body>

<div id="login" class="card">
 <h3>Smart Electrical Fault Detection</h3>
 <input id="u" placeholder="Username"><input id="p" type="password" placeholder="Password">
 <button onclick="login()">Log in</button>
 <button style="background:#5b6b7c" onclick="register()">Register as resident</button>
 <div id="lerr" class="err"></div>
 <div class="hint">Demo: admin/admin123, officer/officer123, resident/resident123</div>
</div>

<div id="app" style="display:none">
 <header><div id="who"></div><button onclick="logout()">Log out</button></header>
 <main>
  <div>
   <div class="card staff"><h3>Live Readings</h3>
    <table><thead><tr><th>Time</th><th>Node</th><th>V</th><th>A</th><th>Leak mA</th><th>PF</th><th>Status</th></tr></thead>
    <tbody id="readings"></tbody></table></div>
   <div class="card"><h3 id="rtitle">Fault Reports</h3>
    <table><thead><tr><th>Time</th><th>User</th><th>Location</th><th>Type</th><th>Description</th></tr></thead>
    <tbody id="reports"></tbody></table></div>
  </div>
  <div>
   <div class="card staff"><h3>Alerts</h3><table><tbody id="alerts"></tbody></table></div>
   <div class="card"><h3>Report a Fault</h3>
    <input id="loc" placeholder="Location (e.g. Kanyama, Block 4)">
    <select id="ftype"><option>Power outage</option><option>Sparking</option><option>Exposed wire</option><option>Other</option></select>
    <textarea id="desc" placeholder="Describe the problem"></textarea>
    <button onclick="sendReport()">Submit Report</button><div id="msg" class="hint"></div></div>
   <div class="card adminonly"><h3>User Management</h3>
    <input id="nu" placeholder="New username"><input id="np" type="password" placeholder="Password (6+ chars)">
    <select id="nr"><option value="safety_officer">Safety officer</option><option value="administrator">Administrator</option><option value="resident">Resident</option></select>
    <button onclick="createUser()">Create user</button><div id="umsg" class="hint"></div>
    <table><tbody id="users"></tbody></table></div>
  </div>
 </main>
</div>

<script>
let token=null, me=null, timer=null;
const nodes=["kanyama-01","matero-01","kanyama-02"];
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

async function api(path,method="GET",body){
 const r=await fetch(path,{method,headers:{"Content-Type":"application/json",...(token?{Authorization:"Bearer "+token}:{})},body:body?JSON.stringify(body):undefined});
 if(r.status===401&&token){logout();return null}
 return r.json();
}
async function login(){
 const r=await fetch("api/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:$("u").value,password:$("p").value})});
 const d=await r.json(); if(!r.ok){$("lerr").textContent=d.error;return}
 token=d.token; me=d; show();
}
async function register(){
 const r=await fetch("api/register",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:$("u").value,password:$("p").value})});
 const d=await r.json(); if(!r.ok){$("lerr").textContent=d.error;return}
 login();
}
function logout(){token=null;me=null;clearInterval(timer);$("app").style.display="none";$("login").style.display="block";$("p").value="";}
function show(){
 $("login").style.display="none"; $("app").style.display="block"; $("lerr").textContent="";
 const staff=me.role!=="resident", admin=me.role==="administrator";
 $("who").textContent=me.username+" ("+me.role.replace("_"," ")+")";
 document.querySelectorAll(".staff").forEach(e=>e.style.display=staff?"block":"none");
 document.querySelectorAll(".adminonly").forEach(e=>e.style.display=admin?"block":"none");
 $("rtitle").textContent=staff?"Resident Fault Reports":"My Reports";
 tick(); timer=setInterval(tick,3000);
}
const row=r=>`<tr><td>${esc(r.time)}</td><td>${esc(r.node_id)}</td><td>${esc(r.voltage)}</td><td>${esc(r.current)}</td><td>${esc(r.leakage_ma)}</td><td>${esc(r.power_factor)}</td><td class="${r.fault==="normal"?"normal":"fault"}">${esc(r.fault)}</td></tr>`;
async function tick(){
 if(!token)return;
 if(me.role!=="resident"){
  await api("api/simulate/"+nodes[Math.floor(Math.random()*nodes.length)]);
  const r=await api("api/readings"), a=await api("api/alerts"); if(!r||!a)return;
  $("readings").innerHTML=r.map(row).join("");
  $("alerts").innerHTML=a.map(x=>`<tr><td>${esc(x.time)}</td><td>${esc(x.node_id)}</td><td class="fault">${esc(x.fault)}</td><td>${x.status==="open"?`<button class="sm" onclick="ack('${esc(x.id)}')">Acknowledge</button>`:"✓ "+esc(x.by)}</td></tr>`).join("")||"<tr><td>No alerts</td></tr>";
 }
 const rp=await api("api/reports"); if(rp)
  $("reports").innerHTML=rp.map(x=>`<tr><td>${esc(x.time)}</td><td>${esc(x.username)}</td><td>${esc(x.location)}</td><td>${esc(x.type)}</td><td>${esc(x.description)}</td></tr>`).join("")||"<tr><td colspan=5>No reports yet</td></tr>";
 if(me.role==="administrator"){const us=await api("api/users"); if(us)$("users").innerHTML=us.map(x=>`<tr><td>${esc(x.username)}</td><td>${esc(x.role)}</td></tr>`).join("");}
}
async function ack(id){await api("api/alerts/"+id+"/ack","POST");tick();}
async function sendReport(){
 const d=await api("api/reports","POST",{location:$("loc").value,type:$("ftype").value,description:$("desc").value});
 $("msg").textContent=d.error||"Report submitted.";
 if(!d.error){$("loc").value="";$("desc").value="";}
 tick();
}
async function createUser(){
 const d=await api("api/users","POST",{username:$("nu").value,password:$("np").value,role:$("nr").value});
 $("umsg").textContent=d.error||"User created."; if(!d.error){$("nu").value="";$("np").value="";} tick();
}
</script></body></html>"""

@app.get("/")
def home(): return PAGE
