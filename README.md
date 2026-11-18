# AI-Powered Answer Script Evaluation through Text Summarization

Micro project, Skill Enhancement Lab (Generative AI), Ramaiah Institute of Technology, 2026-27.
Team: Keerthan AV, Mohith V. Guide: Ms. Evangeline D.

## What the system does

A **teacher-assist tool**, not an auto-grader. The teacher uploads one answer schema
(questions + model answers + point-wise marking scheme). Then they upload the students'
soft-copy answer scripts (.docx or .pdf, answers labelled only by question number).
For each answer the system:

1. finds the matching question in the schema,
2. summarises long answers (gist for the teacher),
3. checks every marking point against the student's sentences (semantic similarity),
4. suggests marks (full / half / zero per point) and totals them,
5. gets a second opinion from a DNN and flags answers where the two disagree,
6. writes short feedback (missed points),
7. stores everything in SQLite so the teacher can review, override and finalise later.

The teacher always makes the final decision.

## Scope

- Soft-copy answers only (no OCR / handwriting).
- Descriptive questions are graded semantically. MCQs use exact match. Numericals are tagged `manual`.
- CBSE-style point-wise marking schemes, with half marks.

## Pipeline

| Phase | Tool / model | Syllabus link |
|---|---|---|
| Parse schema + scripts | python-docx, pdfplumber, regex | - |
| Sentence split | NLTK | NLP basics |
| Summarise (> 80 words) | `sshleifer/distilbart-cnn-12-6` | Hugging Face, Transformers |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Word embeddings, Transformers |
| Point-wise marking | cosine similarity, two thresholds | core idea |
| Second opinion | Keras MLP trained on Mohler | DNNs, TensorFlow |
| Feedback | template; optional LangChain + Cohere + Pydantic | LangChain |
| App | FastAPI + Jinja2 HTML/CSS + SQLAlchemy/SQLite | - |

Point matching always runs on the **full** answer. The summary is for the teacher's quick
view and for the with/without-summarisation experiment.

## Input formats

**Answer schema (teacher)**: see `data/samples/sample_answer_schema.docx`
```
Q1b [2 marks] [type: descriptive]
Question: What is photosynthesis?
Model answer: ...
Points:
- Process by which green plants make their own food (1)
- Uses sunlight, carbon dioxide and water to produce glucose and oxygen (1)
```
Types: `descriptive`, `mcq` (with `Answer: (b)`), `manual`. Internal choice: `Q4-OR`.

**Student script**: any .docx/.pdf where each answer starts with its number:
`Q1 (a)`, `1.b)`, `2)`, `Q3.`, `4 OR`. See `data/samples/student_*.docx`.

## Datasets

- **Mohler ASAG** (`Viswa45/ASAG` on Hugging Face): 2,273 graded open-ended answers, 0-5. Trains the DNN.
- **ASAG2024** (`Meyerger/ASAG2024`): optional extra training data.
- **Own CBSE test set**: our schema + 15-20 classmates' scripts, marked by hand by two people.
  About 5 scripts tune the thresholds, the rest are for testing.

## Evaluation plan

MAE, QWK, Pearson, % within 0.5 marks against human marks, for:
(a) cosine-to-model-answer baseline, (b) point-wise, (c) point-wise + DNN flags,
(d) point-wise with summarisation. Plus a small timing study (minutes per script, with vs. without the tool).

## Folder structure

```
answer-eval/
├── config.py            paths, model names, thresholds
├── requirements.txt
├── data/
│   ├── raw/             downloaded datasets (git-ignored)
│   ├── processed/       features, splits (git-ignored)
│   ├── cbse/schema/     our CBSE answer schema
│   ├── cbse/scripts/    classmates' answer scripts + human marks
│   └── samples/         example schema + student files
├── models/              trained DNN (git-ignored)
├── reports/figures/     plots for the paper
├── scripts/             one-off scripts (download, explore, checks)
├── src/                 grading pipeline (parsers, similarity, dnn, ...)
├── app/                 FastAPI app, templates/, static/
└── tests/
```

## Setup

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py      # Mohler -> data/raw/mohler.csv
python scripts/download_models.py    # caches MiniLM + DistilBART, runs a quick test
python scripts/make_samples.py       # example .docx files
python scripts/explore_mohler.py     # stats + figures
python scripts/check_setup.py        # everything green -> Phase 2
```

## Phases

1. Setup, datasets, models, samples  ← done when check_setup passes
2. Parsers (schema + student scripts)
3. Core grading (embeddings, point-wise marking, baseline, threshold tuning)
4. DNN second opinion (train on Mohler) + flags
5. Summarisation + ablation
6. Database + FastAPI app
7. Feedback (template, then optional LLM)
8. Evaluation on CBSE test set + timing study
9. Demo, documentation, paper results
