# GenSlide — Agentic PowerPoint Generation

> Text and document to PowerPoint slide generation powered by a **LangGraph agentic pipeline** and **GPT-4o**.

GenSlide has been rebuilt from a single-LLM script into a fully agentic system. A multi-node LangGraph graph orchestrates the full pipeline — from parsing your input to building a polished `.pptx` — with a human-in-the-loop review step before finalising the deck.

---

## How it works

```
Input (text / PDF / DOCX)
    └─► Input parser node
            └─► Orchestrator agent    ← GPT-4o plans the slide outline
                    └─► Content agent  ← GPT-4o writes each slide (parallel)
                            └─► PPTX builder node
                                    └─► ⏸ Human approval   ← review in Streamlit
                                            ├─► Approved → save .pptx
                                            └─► Revise   → loop back to orchestrator
```

The **orchestrator** decides how many slides are needed and what order they should go in. The **content agent** fills each slide with bullets and speaker notes, making all GPT-4o calls concurrently so a 10-slide deck generates in roughly the same time as one slide. After every run, you review the deck in the Streamlit UI and either approve it or send targeted revision notes — which the agents use to make surgical edits rather than starting from scratch.

---

## Project structure

```
GenSlide/
├── agents/
│   ├── orchestrator.py      # GPT-4o planner — produces slide outline
│   └── content_agent.py     # GPT-4o writer — fills slides (parallel calls)
├── graph/
│   ├── state.py             # AgentState TypedDict (shared pipeline memory)
│   ├── graph.py             # LangGraph StateGraph — wires nodes + compiles
│   └── human_approval.py    # Interrupt pass-through node
├── tools/
│   ├── input_parser.py      # Plain text, PDF, and DOCX ingestion
│   └── pptx_builder.py      # python-pptx deck assembler
├── frontend/
│   └── ui.py                # Streamlit UI with interrupt/resume flow
├── data/
│   └── content.txt          # Sample input text
├── requirements.txt
└── .env                     # OPENAI_API_KEY goes here
```

---

## Setup

### Prerequisites

- Python 3.10 or higher (not 3.9.7 — `streamlit` does not work on that version)
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to `gpt-4o`

### Install

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/mehdimo/GenSlide
cd GenSlide

python -m venv ./venv
source ./venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...
```

Or export it directly in your shell:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Run

From the project root:

```bash
streamlit run frontend/ui.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Usage

1. **Provide your content** — paste plain text, or upload a PDF or DOCX file.
2. **Click Generate** — the pipeline runs automatically: parsing → planning → writing → building.
3. **Review the draft** — the slide outline and previews appear in the UI. Download the `.pptx` to inspect it in PowerPoint.
4. **Approve or revise** — click **Approve** to finalise, or type revision notes (e.g. *"Add a pricing slide, make the intro shorter"*) and click **Revise**. The agents incorporate your feedback and regenerate.
5. **Download the final deck** — saved to `frontend/generated/` with the iteration number and timestamp in the filename.

---

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Output

Generated decks are saved to `frontend/generated/` and follow this naming convention:

```
deck_v1_20260312_143022.pptx
     ^  ^
     |  timestamp
     iteration number
```

Each deck includes:

- A **title slide** with the deck name and generation date
- **Content slides** with 3–5 bullet points and speaker notes per slide
- A **closing slide**

The visual theme is "Midnight Executive" — navy dominant background, ice-blue accents, pale content slides — designed to look professional out of the box.

---

## Extending the system

The pipeline is designed to be extended one node at a time. Some natural next steps:

- **Research agent** — add a web-search node between the orchestrator and content agent to ground slides in current facts
- **Design agent** — add a node that selects layouts, color themes, or pulls relevant images
- **Critic agent** — add a self-review node that scores the draft and triggers an automatic revision if quality is below a threshold
- **Additional input sources** — the `input_parser` node can be extended to accept URLs, Google Docs, or Notion pages

---

## License

[MIT](LICENSE.md)