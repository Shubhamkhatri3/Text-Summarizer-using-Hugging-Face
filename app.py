# ================================
# 📌 IMPORTS
# ================================

from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re
import PyPDF2
import io

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# ================================
# 📌 APP INIT
# ================================

app = FastAPI(title="AI Summarizer Pro", version="7.0")

# ================================
# 📌 MODEL
# ================================

# 📌 Using a single general-purpose model for both dialogue and
# paragraph/PDF input, to keep the app simple (one model to load, one
# model to explain/debug). facebook/bart-large-cnn is a BART-large model
# fine-tuned on CNN/DailyMail news articles — it's the standard,
# well-tested choice for general document/paragraph summarization, and
# handles short dialogue-style input reasonably well too. Downloaded from
# the Hub on first startup (~1.6GB, cached after that).
MODEL_NAME = "facebook/bart-large-cnn"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ================================
# 📌 TEMPLATE
# ================================

templates = Jinja2Templates(directory="templates")

# ================================
# 📌 REQUEST SCHEMA
# ================================

class DialogueInput(BaseModel):
    dialogue: str
    mode: str = "detailed"  # short / detailed / bullets

# ================================
# 📌 CLEAN
# ================================

def clean_data(text):
    return re.sub(r"\s+", " ", text).strip()

def clean_summary(text):
    """
    Safety net for a failure mode where the model, pushed to a longer
    min_length than the input really supports, invents a trailing
    URL-like fragment (e.g. '..org.uk/Some-Made-Up-Path.') that wasn't in
    the source text at all. Strips just that fragment, keeping the real
    word/sentence before it intact.
    """
    text = re.sub(r"\.{1,2}(?:[\w-]*\.)?(?:com|org|net|gov|edu|co|uk|io)(?:\.\w+)?/[\w\-/]*\.?", ".", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()

# ================================
# 📌 MODE PROMPT
# ================================

def build_prompt(text, mode):
    # 📌 The old T5 checkpoint was trained to respond to instruction-style
    # prefixes like "summarize briefly:". BART-SAMSum was NOT trained that
    # way — it expects the raw dialogue/text with no prefix, and adding one
    # would just confuse it. Mode is now handled via generation length in
    # summarize_text() instead of via the prompt text.
    return text

# ================================
# 📌 CORE SUMMARIZER
# ================================

def summarize_text(text, mode="detailed"):

    prompt = build_prompt(text, mode)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 📌 Mode now controls length directly via generation params, since this
    # model doesn't respond to text-prefix instructions like the old one did.
    input_len = inputs["input_ids"].shape[1]

    # 📌 Raising max_length alone didn't fix truncated summaries — the model
    # was stopping itself early via early_stopping once min_length was
    # satisfied and it found an end-of-sentence token, regardless of how
    # much max_length room was left. The real fix is scaling min_length
    # up with input size, so it's FORCED to keep generating past the
    # opening sentences and actually cover the back half of longer text.
    if mode == "short":
        dynamic_min = 10
        dynamic_max = max(40, int(input_len * 0.25))
    elif mode == "bullets":
        dynamic_min = max(20, int(input_len * 0.20))
        dynamic_max = max(100, int(input_len * 0.45))
    else:  # detailed
        # 📌 0.30 pushed min_length just past how much real content this
        # kind of paragraph actually has, causing an invented trailing
        # URL fragment. 0.24 is a slightly safer margin; clean_summary()
        # below is the backstop if it still overshoots occasionally.
        dynamic_min = max(30, int(input_len * 0.24))
        dynamic_max = max(130, int(input_len * 0.60))

    # Keep a sane ceiling so it never runs away on very large inputs/chunks
    dynamic_max = min(dynamic_max, 300)
    dynamic_min = min(dynamic_min, dynamic_max - 15)

    outputs = model.generate(
        inputs["input_ids"],
        max_length=dynamic_max,
        min_length=dynamic_min,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return clean_summary(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ================================
# 📌 CHUNKING
# ================================

def split_text(text, chunk_size=400):
    # 📌 Split on sentence boundaries first, then group sentences into
    # chunks close to chunk_size words. The old version split on raw word
    # count, which could cut a sentence in half between two chunks and
    # confuse the model.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_len + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sentence)
        current_len += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ================================
# 📌 LARGE TEXT SUMMARIZATION
# ================================

def summarize_large_text(text, mode):

    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        summaries.append(summarize_text(chunk, mode))

    combined = " ".join(summaries)

    return summarize_text(combined, mode)

# ================================
# 📌 PDF EXTRACTION
# ================================

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    return text

# ================================
# 📌 QUALITY CHECK (rejects resumes/tables/forms)
# ================================

def is_summarizable(text):
    """
    Structured documents like resumes, tables and forms extract as short,
    disconnected lines rather than continuous prose. Feeding that into the
    model is what caused it to hallucinate (e.g. inventing conferences,
    organizations, or job titles that were never actually stated).

    The word-count/sentence-count check alone isn't enough: resume bullet
    points are often full, well-punctuated sentences, so they pass a
    generic prose check while still being fundamentally a resume (a
    person's name, disconnected project/skill entries, no real narrative
    connecting them). So this also checks for resume-specific signals:
    section headers (SKILLS, EDUCATION, etc.) and contact-info patterns
    (email, phone). Either signal on its own is a strong enough indicator
    to reject, regardless of how "prose-like" individual lines look.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False

    avg_words_per_line = sum(len(l.split()) for l in lines) / len(lines)
    sentence_count = len(re.findall(r'[.!?]', text))

    if avg_words_per_line < 6 or sentence_count < 3:
        return False

    resume_headers = [
        "skills", "education", "experience", "projects", "personal projects",
        "achievements", "certifications", "objective", "summary",
        "work experience", "technical skills", "contact"
    ]
    header_hits = sum(
        1 for l in lines
        if l.strip().lower().rstrip(":") in resume_headers
    )

    has_email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text) is not None
    has_phone = re.search(r"(\+?\d[\d\-\s]{8,}\d)", text) is not None

    if header_hits >= 2 or (header_hits >= 1 and (has_email or has_phone)):
        return False

    return True

# ================================
# 📌 TEXT API
# ================================

@app.post("/summarize/")
async def summarize(data: DialogueInput):
    return {
        "summary": summarize_text(data.dialogue, data.mode)
    }

# ================================
# 📌 PDF API
# ================================

@app.post("/summarize-pdf/")
async def summarize_pdf(file: UploadFile = File(...), mode: str = "detailed"):

    contents = await file.read()
    pdf_file = io.BytesIO(contents)

    text = extract_text_from_pdf(pdf_file)

    if not text.strip():
        return {"error": "No text found in PDF"}

    if not is_summarizable(text):
        return {"error": "This PDF looks like a resume, table, or form rather than continuous text, so a reliable summary can't be generated from it. Try a document with full paragraphs (e.g. an article or report)."}

    summary = summarize_large_text(text, mode)

    return {"summary": summary}

# ================================
# 📌 HOME
# ================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request}
    )

# ================================
# 📌 RUN
# ================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
