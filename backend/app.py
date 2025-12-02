import os
import sqlite3
from datetime import datetime
from flask import (
    Flask, request, jsonify, send_from_directory, session,
    redirect, url_for, render_template, abort
)
from werkzeug.utils import secure_filename

# import the two analyzers
from analyzer_control import analyze_csv_control
from analyzer_stroke import analyze_csv_stroke

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "..", "outputs")
DB_PATH = os.path.join(BASE_DIR, "database.db")
ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# templates located in ../frontend/templates
# static located in ../frontend/static
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "frontend", "templates"),
    static_folder=os.path.join(BASE_DIR, "..", "frontend", "static"),
    static_url_path="/static"
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

app.secret_key = "dev-secret-key-change-for-prod"

# --- Hardcoded physiotherapists (dev-only) ---
HARDCODED_PHYSIOS = {
    "Maria Eduarda": "senha123",
    "Arthur": "senha234",
    "Beatriz": "senha345",
    "Anita": "senha456",
    "Fernanda": "senha567",
    "Pedro": "senha678"
}

PHYSIO_DISPLAY = {
    "Maria Eduarda": "Maria Eduarda",
    "Arthur": "Arthur",
    "Beatriz": "Beatriz",
    "Anita": "Anita",
    "Fernanda": "Fernanda",
    "Pedro": "Pedro"
}

# DB helpers
def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_conn()
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS physios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        display_name TEXT
    );

    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        physio_id INTEGER,
        name TEXT,
        dob TEXT,
        group_type TEXT DEFAULT 'controle',
        created_at TEXT,
        FOREIGN KEY(physio_id) REFERENCES physios(id)
    );

    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        physio_id INTEGER,
        csv_filename TEXT,
        image_filename TEXT,
        note TEXT,
        created_at TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(id),
        FOREIGN KEY(physio_id) REFERENCES physios(id)
    );

    /* NOVA TABELA DE FICHA MÉDICA */
    CREATE TABLE IF NOT EXISTS medical_forms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        patient_name TEXT,
        age TEXT,
        gender TEXT,
        diagnosis TEXT,
        symptoms TEXT,
        treatment_start TEXT,
        form_date TEXT,
        qualitative_progress TEXT,
        notes TEXT,
        created_at TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(id)
    );
    """)

    # cria fisioterapeutas hardcoded se não existirem
    for username in HARDCODED_PHYSIOS:
        cur.execute("SELECT id FROM physios WHERE username = ?", (username,))
        if cur.fetchone() is None:
            display = PHYSIO_DISPLAY.get(username, username)
            cur.execute("INSERT INTO physios (username, display_name) VALUES (?, ?)", (username, display))

    conn.commit()
    conn.close()

    # ensure patients table has group_type (for older DBs)
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(patients)")
    cols = [r[1] for r in cur.fetchall()]
    if 'group_type' not in cols:
        try:
            cur.execute("ALTER TABLE patients ADD COLUMN group_type TEXT DEFAULT 'controle'")
        except Exception:
            pass
    conn.commit()
    conn.close()


init_db()

# auth helpers
def login_user(username):
    session["username"] = username
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM physios WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    session["physio_id"] = row["id"] if row else None


def logout_user():
    session.pop("username", None)
    session.pop("physio_id", None)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------ TEMPLATED PAGES ------------------

@app.route("/login")
def login_page():
    # if already logged, redirect
    if "username" in session:
        return redirect(url_for("dashboard_page"))
    return render_template("login.html")


@app.route("/")
def home():
    # redirect appropriately
    if "username" in session:
        return redirect(url_for("dashboard_page"))
    return redirect(url_for("login_page"))


@app.route("/dashboard")
def dashboard_page():
    if "username" not in session:
        return redirect(url_for("login_page"))
    return render_template("dashboard.html", username=session.get("username"))


@app.route("/patients")
def patients_page():
    if "username" not in session:
        return redirect(url_for("login_page"))
    return render_template("patients.html")


@app.route("/patients/<int:patient_id>")
def patient_page(patient_id):
    if "username" not in session:
        return redirect(url_for("login_page"))

    conn = get_db_conn()
    cur = conn.cursor()

    # paciente pertence ao fisio?
    cur.execute("SELECT * FROM patients WHERE id=? AND physio_id=?", (patient_id, session["physio_id"]))
    p = cur.fetchone()

    if not p:
        conn.close()
        abort(404)

    cur.execute("SELECT * FROM medical_forms WHERE patient_id=? ORDER BY created_at DESC", (patient_id,))
    forms = [dict(f) for f in cur.fetchall()]

    conn.close()

    return render_template("patient.html", patient=dict(p), medical_forms=forms)

# serve outputs (images)
@app.route("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, "..", "outputs"), filename)

# ------------------ API endpoints (same behavior as before) ------------------

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    if username in HARDCODED_PHYSIOS and HARDCODED_PHYSIOS[username] == password:
        login_user(username)
        return jsonify({"ok": True, "username": username, "display_name": PHYSIO_DISPLAY.get(username, username)})
    return jsonify({"error": "invalid credentials"}), 401


@app.route("/api/logout", methods=["POST"])
def api_logout():
    logout_user()
    return jsonify({"ok": True})


@app.route("/api/whoami", methods=["GET"])
def api_whoami():
    if "username" in session:
        return jsonify({"username": session["username"], "physio_id": session.get("physio_id")})
    return jsonify({"username": None}), 200

# patients CRUD
@app.route("/api/patients", methods=["GET"])
def api_get_patients():
    if "physio_id" not in session:
        return jsonify({"error": "not logged in"}), 401
    physio_id = session["physio_id"]
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE physio_id = ? ORDER BY created_at DESC", (physio_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.route("/api/patients", methods=["POST"])
def api_create_patient():
    if "physio_id" not in session:
        return jsonify({"error": "not logged in"}), 401
    data = request.json or {}
    name = data.get("name", "").strip()
    dob = data.get("dob", "").strip()
    create_ficha = bool(data.get("create_ficha", False))
    group_type = data.get("group_type", "controle")  # expected 'controle' or 'avc'
    if group_type not in ("controle", "avc"):
        group_type = "controle"
    if not name:
        return jsonify({"error": "patient name required"}), 400
    physio_id = session["physio_id"]
    conn = get_db_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO patients (physio_id, name, dob, group_type, created_at) VALUES (?, ?, ?, ?, ?)", (physio_id, name, dob, group_type, now))
    pid = cur.lastrowid
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "patient_id": pid, "create_ficha": create_ficha}), 201


@app.route("/api/patients/<int:patient_id>", methods=["DELETE"])
def api_delete_patient(patient_id):
    if "physio_id" not in session:
        return jsonify({"error": "not logged in"}), 401
    physio_id = session["physio_id"]
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM patients WHERE id = ? AND physio_id = ?", (patient_id, physio_id))
    if cur.fetchone() is None:
        conn.close()
        return jsonify({"error": "patient not found or not owned"}), 404
    cur.execute("DELETE FROM analyses WHERE patient_id = ?", (patient_id,))
    cur.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

# upload + analysis
@app.route("/upload", methods=["POST"])
def upload_file():
    if "physio_id" not in session:
        return jsonify({"error": "not logged in"}), 401
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400
    patient_id = request.form.get("patient_id")
    if not patient_id:
        return jsonify({"error": "patient_id is required (select a patient)"}), 400
    # verify ownership
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, group_type FROM patients WHERE id = ? AND physio_id = ?", (patient_id, session["physio_id"]))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return jsonify({"error": "patient not found or not owned by you"}), 403

    group_type = row["group_type"] if isinstance(row, dict) else (row[1] if row and len(row) > 1 else 'controle')

    file = request.files["file"]
    if file.filename == "":
        conn.close()
        return jsonify({"error": "Nome do arquivo vazio."}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_name = f"{timestamp}_{filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
        file.save(save_path)
        try:
            # choose analyzer based on patient group_type
            if group_type == 'avc':
                out_image_path, stats = analyze_csv_stroke(save_path, app.config["OUTPUT_FOLDER"])
            else:
                out_image_path, stats = analyze_csv_control(save_path, app.config["OUTPUT_FOLDER"])

            now = datetime.utcnow().isoformat()
            image_basename = os.path.basename(out_image_path)
            cur.execute(
                "INSERT INTO analyses (patient_id, physio_id, csv_filename, image_filename, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (patient_id, session["physio_id"], save_name, image_basename, "", now)
            )
            conn.commit()
            analysis_id = cur.lastrowid
            conn.close()
            image_url = f"/outputs/{image_basename}"
            return jsonify({"image_url": image_url, "stats": stats, "analysis_id": analysis_id}), 200
        except Exception as e:
            conn.close()
            return jsonify({"error": "Erro ao processar CSV.", "details": str(e)}), 500
    conn.close()
    return jsonify({"error": "Tipo de arquivo não suportado. Envie CSV."}), 400


@app.route("/api/patients/<int:patient_id>/analyses", methods=["GET"])
def api_patient_analyses(patient_id):
    if "physio_id" not in session:
        return jsonify({"error": "not logged in"}), 401
    physio_id = session["physio_id"]
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE id = ? AND physio_id = ?", (patient_id, physio_id))
    if cur.fetchone() is None:
        conn.close()
        return jsonify({"error": "patient not found or not owned"}), 404
    cur.execute("SELECT * FROM analyses WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.route("/api/analyses/<int:analysis_id>", methods=["DELETE"])
def api_delete_analysis(analysis_id):
    if "physio_id" not in session:
        return jsonify({"error": "not logged in"}), 401
    physio_id = session["physio_id"]
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM analyses WHERE id = ? AND physio_id = ?", (analysis_id, physio_id))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return jsonify({"error": "analysis not found or not owned"}), 404
    image_filename = row["image_filename"]
    cur.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()
    im_path = os.path.join(app.config["OUTPUT_FOLDER"], image_filename)
    if os.path.exists(im_path):
        try:
            os.remove(im_path)
        except:
            pass
    return jsonify({"ok": True})

@app.route("/patients/<int:patient_id>/medical-form/new", methods=["GET"])
def medical_form_new(patient_id):
    if "physio_id" not in session:
        return redirect("/login")

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE id=? AND physio_id=?", (patient_id, session["physio_id"]))
    patient = cur.fetchone()
    conn.close()

    if not patient:
        abort(404)

    return render_template("medical_form_new.html", patient=patient)

@app.route("/patients/<int:patient_id>/medical-form/new", methods=["POST"])
def medical_form_create(patient_id):
    if "physio_id" not in session:
        return redirect("/login")

    f = request.form
    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO medical_forms (
            patient_id, patient_name, age, gender, diagnosis, symptoms,
            treatment_start, form_date, qualitative_progress, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        f.get("patient_name"), f.get("age"), f.get("gender"),
        f.get("diagnosis"), f.get("symptoms"),
        f.get("treatment_start"), f.get("form_date"),
        f.get("qualitative_progress"), f.get("notes"),
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()

    return redirect(f"/patients/{patient_id}")

@app.route("/patients/<int:patient_id>/medical-form/<int:form_id>")
def medical_form_view(patient_id, form_id):
    if "physio_id" not in session:
        return redirect("/login")

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM medical_forms WHERE id=? AND patient_id=?", (form_id, patient_id))
    form = cur.fetchone()
    conn.close()

    if not form:
        abort(404)

    return render_template("medical_form_view.html", form=form)

@app.route("/patients/<int:patient_id>/medical-form/<int:form_id>/edit", methods=["GET"])
def medical_form_edit(patient_id, form_id):
    if "physio_id" not in session:
        return redirect("/login")

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM medical_forms WHERE id=? AND patient_id=?", (form_id, patient_id))
    form = cur.fetchone()
    conn.close()

    if not form:
        abort(404)

    return render_template("medical_form_edit.html", form=form)

@app.route("/patients/<int:patient_id>/medical-form/<int:form_id>/edit", methods=["POST"])
def medical_form_update(patient_id, form_id):
    if "physio_id" not in session:
        return redirect("/login")

    f = request.form
    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE medical_forms
        SET patient_name=?, age=?, gender=?, diagnosis=?, symptoms=?,
            treatment_start=?, form_date=?, qualitative_progress=?, notes=?
        WHERE id=? AND patient_id=?
    """, (
        f.get("patient_name"), f.get("age"), f.get("gender"),
        f.get("diagnosis"), f.get("symptoms"),
        f.get("treatment_start"), f.get("form_date"),
        f.get("qualitative_progress"), f.get("notes"),
        form_id, patient_id
    ))

    conn.commit()
    conn.close()

    return redirect(f"/patients/{patient_id}")

if __name__ == "__main__":
    # dev only
    app.run(host="0.0.0.0", port=5000, debug=True)