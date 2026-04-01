# ================================
# 📌 IMPORTS
# ================================

from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
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

model = T5ForConditionalGeneration.from_pretrained("./saved_summary_modell")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_modell")

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

# ================================
# 📌 MODE PROMPT
# ================================

def build_prompt(text, mode):

    if mode == "short":
        return "summarize briefly: " + text

    elif mode == "bullets":
        return "summarize in bullet points: " + text

    return "summarize in detail: " + text

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

    outputs = model.generate(
        inputs["input_ids"],
        max_length=312,
        min_length=72,
        num_beams=6,
        length_penalty=2.7,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ================================
# 📌 CHUNKING
# ================================

def split_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

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