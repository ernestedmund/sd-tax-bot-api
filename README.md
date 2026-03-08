# SD County Property Tax Assistant — RAG Architecture

## What changed and why

### Before (expensive)
```
Browser → Anthropic API directly
         └─ Sends entire 50KB knowledge base every message
         └─ Uses Claude Sonnet (~$3/M tokens)
         └─ API key exposed in browser source code
         └─ ~$0.045 per question
```

### After (cheap + secure)
```
Browser → Your FastAPI server → Anthropic API
          └─ RAG retrieves 3-4 relevant chunks only (~3KB)
          └─ Uses Claude Haiku ($0.80/M input, $4/M output)
          └─ API key lives on your server only
          └─ Rate limiting per IP
          └─ ~$0.0004 per question (100x cheaper)
```

---

## File overview

```
sd-rag/
├── rag_engine.py   # Core RAG logic: KB parsing, TF-IDF indexing, retrieval
├── server.py       # FastAPI wrapper: REST endpoint, rate limiting, sessions
├── index.html      # Frontend: calls your server instead of Anthropic directly
└── README.md       # This file
```

---

## How RAG works here

1. **Chunking** — The knowledge base is split into individual topic entries
   (P13-001, CIO-004, etc.). Each chunk is ~150-300 words.

2. **Indexing** — At server startup, each chunk is converted to a TF-IDF
   vector (a sparse representation of which words appear and how important
   they are). This takes ~50ms and happens once.

3. **Retrieval** — When a user asks a question, the question is also
   converted to a TF-IDF vector, and cosine similarity is computed against
   all chunks. The top 4 most relevant chunks are selected.

4. **Prompting** — Only those 4 chunks (~3KB) are sent to Claude Haiku,
   not the full knowledge base. The system prompt instructs the model to
   answer only from what it's given.

5. **Response** — Claude answers using only the retrieved context.

---

## Cost model

| Scenario              | Tokens/query | Cost/query  | 10K queries/mo |
|-----------------------|-------------|-------------|----------------|
| Old (Sonnet, full KB) | ~15,000     | ~$0.045     | ~$450          |
| New (Haiku, RAG)      | ~1,500      | ~$0.0004    | ~$4            |

Haiku pricing (2025): $0.80/M input tokens, $4.00/M output tokens.

At 10,000 questions/month (generous for a county tool), cost is ~$4/month.
Even at 100,000 questions/month: ~$40/month.

---

## Upgrading from TF-IDF to real embeddings

The TF-IDF retriever works well for this small, structured KB.
For a larger KB or better semantic matching, swap it out:

### Option A: OpenAI embeddings (best quality, tiny cost)
```python
from openai import OpenAI
openai_client = OpenAI()

def embed(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding
```
Cost: ~$0.00002 per query. For 100K queries/month = $2.

### Option B: Local embeddings (free, runs on county server)
```bash
pip install sentence-transformers
```
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, downloads once

def embed(text: str) -> list[float]:
    return model.encode(text).tolist()
```
No API cost. Requires ~500MB RAM. Works offline.

For either option, pre-embed all chunks at startup, store as numpy arrays,
and use numpy dot product instead of the cosine_similarity function.

---

## Deployment on Azure (likely your existing IT contract)

### Azure App Service (simplest)
```bash
# Install Azure CLI, then:
az login
az group create --name sd-tax-bot --location westus
az appservice plan create --name sd-tax-plan --resource-group sd-tax-bot --sku B1 --is-linux
az webapp create --name sd-tax-assistant --resource-group sd-tax-bot \
  --plan sd-tax-plan --runtime "PYTHON:3.11"

# Set your API key as an environment variable (never in code)
az webapp config appsettings set --name sd-tax-assistant \
  --resource-group sd-tax-bot \
  --settings ANTHROPIC_API_KEY="sk-ant-..."

# Deploy
az webapp up --name sd-tax-assistant
```

B1 tier costs ~$13/month. More than enough for this traffic.

### Update the frontend
In `index.html`, change:
```js
const API_BASE = 'https://YOUR-BACKEND-URL.azurewebsites.net';
```
to:
```js
const API_BASE = 'https://sd-tax-assistant.azurewebsites.net';
```

### Update CORS in server.py
```python
ALLOWED_ORIGINS = [
    "https://YOUR-ORG.github.io",  # your GitHub Pages URL
]
```

---

## Rate limiting

Current: 15 questions per IP per hour (in-memory).

For production at county scale, use Redis:
```bash
pip install redis
```
```python
import redis
r = redis.Redis(host='your-redis-host', port=6379)

def check_rate_limit(ip: str):
    key = f"ratelimit:{ip}"
    count = r.incr(key)
    if count == 1:
        r.expire(key, 3600)  # 1 hour TTL
    if count > 15:
        raise HTTPException(status_code=429, detail="Rate limit reached.")
```
Azure Cache for Redis starts at ~$16/month.

---

## Adding more knowledge base content

Each KB entry follows this format in `rag_engine.py`:

```
## TOPIC-CODE | Category Name | Question being answered
Answer text here. Can be multiple sentences.
Keep each entry focused on one question/concept.
Source: BOE Publication XX · TOPIC-CODE
```

Add entries to the `RAW_KNOWLEDGE_BASE` string in `rag_engine.py`.
The server auto-indexes them at startup — no other changes needed.

---

## Security checklist for county IT

- [x] API key stored as server environment variable, not in code
- [x] API key never sent to browser
- [x] CORS restricted to your GitHub Pages domain
- [x] Input length limited to 1000 characters
- [x] Rate limiting per IP (15/hour)
- [x] No PII collected — no parcel numbers, names, or account data stored
- [x] Session history cleared on page reload
- [ ] Add HTTPS (automatic with Azure App Service)
- [ ] Consider IP allowlist if tool is internal-only
- [ ] Log queries to Azure Monitor for usage analytics

---

## Monthly cost summary (estimate)

| Item                          | Cost/month |
|-------------------------------|------------|
| Azure App Service (B1)        | ~$13       |
| Anthropic API (100K queries)  | ~$40       |
| Azure Redis (optional)        | ~$16       |
| **Total**                     | **~$70**   |

Compare to: $0.045 × 100,000 = $4,500/month with the original setup.
