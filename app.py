import os
import re
import random
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from PyPDF2 import PdfReader
from docx import Document as WordDocument
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

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

# =============================
# UTIL
# =============================
def extract_text_from_file(filepath):
    text = ""
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    elif filepath.endswith(".docx"):
        doc = WordDocument(filepath)
        for p in doc.paragraphs:
            text += p.text + "\n"
    return text


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [STEMMER.stem(w) for w in tokens if w not in STOPWORDS and len(w) > 3]
    return " ".join(tokens)


def analyze_text(text):
    sentences = sent_tokenize(text)
    clean = [preprocess_text(s) for s in sentences]
    if not clean:
        return []

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(clean)
    scores = tfidf_matrix.sum(axis=1).A1

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:10]]


def generate_questions(sentences, jumlah, jenis):
    hasil = []
    random.shuffle(sentences)

    for kalimat in sentences:
        words = [w for w in kalimat.split() if len(w) > 4]
        if not words:
            continue

        if jenis == "pilihan_ganda":
            jawab = random.choice(words)
            soal = kalimat.replace(jawab, "_____")
            opsi = list(set(random.sample(words, min(4, len(words)))))
            hasil.append({
                "question": soal,
                "options": opsi,
                "answer": jawab,
                "type": "pg"
            })
        else:
            hasil.append({
                "question": f"Jelaskan maksud dari pernyataan berikut: {kalimat}",
                "answer": kalimat,
                "type": "essay"
            })

        if len(hasil) >= jumlah:
            break

    return hasil


# =============================
# ROUTES
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
    global questions_global, jenis_soal_global

    if request.method == "POST":
        file = request.files["file"]
        jenis_soal_global = request.form["jenis_soal"]
        jumlah = int(request.form["jumlah_soal"])

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        text = extract_text_from_file(path)
        sentences = analyze_text(text)

        questions_global = generate_questions(sentences, jumlah, jenis_soal_global)
        return redirect(url_for("result"))

    return render_template("index.html")


@app.route("/result")
def result():
    return render_template(
        "result.html",
        questions=questions_global,
        jenis_soal=jenis_soal_global
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


if __name__ == "__main__":
    app.run(debug=True)
