import os
import re
import random
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from PyPDF2 import PdfReader
from docx import Document as WordDocument
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from db import get_db

# ===== NLP =====
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

STOPWORDS = set(stopwords.words("indonesian"))
STEMMER = StemmerFactory().create_stemmer()

questions_global = []
jenis_soal_global = ""
materi_global = ""

# =============================
# UTIL FILE
# =============================
def extract_text_from_file(filepath):
    text = ""
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    elif filepath.endswith(".docx"):
        doc = WordDocument(filepath)
        for p in doc.paragraphs:
            text += p.text + "\n"
    return text

# =============================
# NLP
# =============================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [STEMMER.stem(w) for w in tokens if w not in STOPWORDS and len(w) > 3]
    return " ".join(tokens)

def analyze_text(text):
    sentences = sent_tokenize(text)
    sentences = [s for s in sentences if len(s.split()) > 8]

    if not sentences:
        return []

    clean = [preprocess_text(s) for s in sentences]

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(clean)
    scores = matrix.sum(axis=1).A1

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:15]]

def is_kalimat_layak(kalimat):
    if len(kalimat.split()) < 10:
        return False

    kata_ambigu = ["ini", "itu", "tersebut", "diatas", "dibawah"]
    if any(k in kalimat.lower() for k in kata_ambigu):
        return False

    return True

# =============================
# ESSAY
# =============================
def ambil_konsep(kalimat):
    stop_konsep = ["manfaat", "contoh", "proses", "hasil", "konsep","tersebut","ini","itu"]

    words = kalimat.lower().replace(".", "").split()
    kandidat = [w for w in words if len(w) > 5 and w not in stop_konsep]

    return kandidat[0] if kandidat else None

def generate_essay_answer(kalimat, tipe):
    if tipe == "why":
        return (
            "Karena " + kalimat.lower()
            + " sehingga konsep tersebut penting untuk dipahami "
              "dalam konteks pembelajaran."
        )

    return (
        kalimat[0].upper()
        + kalimat[1:]
        + ". Hal ini menunjukkan bahwa konsep tersebut memiliki peranan penting "
          "dalam pembahasan materi."
    )

def generate_essay_questions(sentences, jumlah):
    hasil = []
    konsep_dipakai = set()

    for kalimat in sentences:
        if not is_kalimat_layak(kalimat):
            continue

        konsep = ambil_konsep(kalimat)
        if not konsep or konsep in konsep_dipakai:
            continue

        konsep_dipakai.add(konsep)
        kl = kalimat.lower()

        if "adalah" in kl:
            tipe = "definisi"
        elif "fungsi" in kl:
            tipe = "fungsi"
        elif "proses" in kl:
            tipe = "proses"
        else:
            tipe = "why"

        kata_kerja = KATA_KERJA_KOGNITIF[tipe]
        pertanyaan = f"{kata_kerja} {konsep} berdasarkan materi di atas."

        hasil.append({
            "question": pertanyaan,
            "answer": generate_essay_answer(kalimat, tipe),
            "materi": kalimat,
            "type": "essay",
            "quality": "validated"
        })

        if len(hasil) >= jumlah:
            break

    return hasil

# =============================
# PILIHAN GANDA KONSEP
# =============================
KONSEP_RULES = {
    "penyebab": [
        "disebab",
        "karena",
        "dipengaruhi"
    ],
    "fungsi": [
        "fungsi",
        "berfungsi",
        "digunakan",
        "berperan"
    ],
    "akibat": [
        "akibat",
        "mengakibatkan",
        "menyebabkan",
        "berdampak"
    ]
}

KATA_KERJA_KOGNITIF = {
    "definisi": "Jelaskan",
    "fungsi": "Uraikan",
    "proses": "Jabarkan",
    "why": "Analisis"
}

def deteksi_konsep(kalimat):
    teks = kalimat.lower()
    for konsep, pola_list in KONSEP_RULES.items():
        for pola in pola_list:
            if pola in teks:
                return konsep
    return None

def generate_pg_questions(sentences, jumlah):
    hasil = []
    KONSEP_JAWABAN = list(KONSEP_RULES.keys())

    # PASS 1 – rule semantik
    for kalimat in sentences:
        konsep = deteksi_konsep(kalimat)
        if not konsep:
            continue

        distraktor = list(set(KONSEP_JAWABAN) - {konsep})
        random.shuffle(distraktor)

        opsi = [konsep] + distraktor[:3]
        random.shuffle(opsi)

        hasil.append({
            "question": (
                "Pernyataan:\n\n"
                f"{kalimat}\n\n"
                "apa yang dimaksud pada pernyataan tersebut …"
            ),
            "options": opsi,
            "answer": konsep,
            "type": "pg"
        })

        if len(hasil) >= jumlah:
            return hasil

    # PASS 2 – fallback keyword
    for kalimat in sentences:
        if len(hasil) >= jumlah:
            break

        kata = ambil_konsep(kalimat)
        if not kata:
            continue

        distraktor = random.sample(KONSEP_JAWABAN, min(3, len(KONSEP_JAWABAN)))
        opsi = [kata] + distraktor
        random.shuffle(opsi)

        hasil.append({
            "question": (
                "Pernyataan:\n\n"
                f"{kalimat}\n\n"
                "pengertian yang sesuai dari pernyataan tersebut yaitu …"
            ),
            "options": opsi,
            "answer": kata,
            "type": "pg"
        })

    return hasil

# =============================
# ROUTER UTAMA
# =============================
def generate_questions(sentences, jumlah, jenis):
    if jenis == "essay":
        return generate_essay_questions(sentences, jumlah)
    else:
        return generate_pg_questions(sentences, jumlah)

# DATABASE
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS materi (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        isi TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS soal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pertanyaan TEXT,
        jawaban TEXT,
        jenis TEXT,
        materi_id INTEGER
    )
    """)

    conn.commit()
    conn.close()

# =============================
# ROUTES
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
    global questions_global, jenis_soal_global, materi_global

    if request.method == "POST":
        file = request.files["file"]
        jenis_soal_global = request.form["jenis_soal"]
        jumlah = int(request.form["jumlah_soal"])

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # Extract materi
        materi_global = extract_text_from_file(path)

        # Simpan ke DB
        conn = get_db()
        c = conn.cursor()
        c.execute("INSERT INTO materi (isi) VALUES (?)", (materi_global,))
        materi_id = c.lastrowid
        conn.commit()
        conn.close()

        # Analisis teks & generate soal
        sentences = analyze_text(materi_global)
        questions_global = generate_questions(sentences, jumlah, jenis_soal_global)

        return redirect(url_for("result"))

    return render_template("index.html")

@app.route("/result")
def result():
    global materi_global, questions_global, jenis_soal_global

    return render_template(
        "result.html",
        questions=questions_global,
        jenis_soal=jenis_soal_global,
        materi=materi_global
    )

@app.route("/update-answer", methods=["POST"])
def update_answer():
    index = int(request.form["index"])
    new_answer = request.form["answer"]

    questions_global[index]["answer"] = new_answer
    questions_global[index]["edited"] = True

    return redirect(url_for("result"))

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

@app.route("/export-pdf")
def export_pdf():
    filename = "soal_otomatis.pdf"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "SOAL OTOMATIS")
    y -= 30

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Jenis Soal: {jenis_soal_global}")
    y -= 30

    for i, q in enumerate(questions_global, 1):
        if y < 100:
            c.showPage()
            y = height - 50

        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, f"{i}. {q['question']}")
        y -= 18

        if q["type"] == "pg":
            for opt in q["options"]:
                c.setFont("Helvetica", 10)
                c.drawString(60, y, f"- {opt}")
                y -= 14

        c.setFont("Helvetica-Oblique", 10)
        c.drawString(60, y, f"Jawaban: {q['answer']}")
        y -= 25

    c.save()
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
