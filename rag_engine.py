"""
SD County Property Tax Assistant - RAG Engine
=============================================
Chunks the knowledge base, builds a TF-IDF vector index (no external
embedding API needed), retrieves relevant chunks at query time, and
calls Claude Haiku with only those chunks.

For production: swap TF-IDF for a real embedding model (see comments).
"""

import json
import re
import math
import os
from collections import Counter
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 1. KNOWLEDGE BASE
#    Each entry maps directly to a topic code in your existing KB.
#    Format: { "id": "P13-001", "topic": "...", "content": "..." }
# ---------------------------------------------------------------------------

RAW_KNOWLEDGE_BASE = """
## P13-001 | Prop 13 Basics | What is Proposition 13?
Proposition 13 is a California law passed by voters in 1978 that limits how much your property's taxable value can increase each year. When you buy a property or complete new construction, a "base year value" is set — and that value can only go up by a maximum of 2% per year, no matter what the real estate market does.
Source: BOE Publication 29 · P13-001

## P13-002 | Prop 13 Basics | What is a base year value?
Your base year value is the taxable value assigned to your property when it is purchased or when new construction is completed. For a purchase, it is generally equal to the sale price. This becomes the starting point for your property taxes, and it can only increase by up to 2% per year going forward.
Source: BOE Publication 29 · P13-002

## P13-003 | Prop 13 Basics | What is the property tax rate in San Diego County?
The base property tax rate in California is 1% of your property's assessed value. However, your total tax bill is typically higher than 1% because it also includes voter-approved charges such as school bonds, Mello-Roos special taxes, and other local assessments that vary by location.
Source: SD County Treasurer-Tax Collector · P13-003

## P13-004 | Prop 13 Basics | Why did my assessed value go up?
Each year the county may increase your assessed value by up to 2% for inflation. If it went up more than 2%, a Change in Ownership or new construction likely triggered a reassessment to current market value. You have the right to appeal if you believe the value is incorrect.
Source: BOE Publication 29 · P13-004

## P13-005 | Prop 13 Basics | What triggers a reassessment to full market value?
Two events trigger a reassessment: a Change in Ownership (such as a sale) and the completion of new construction. Outside of these events, assessed value can only increase by up to 2% per year.
Source: BOE Publication 29 · P13-005

## P13-006 | Prop 13 Basics | Can my assessed value go down? What is Prop 8?
Yes. If current market value falls below your Prop 13 assessed value, the county must temporarily reduce your assessed value. This is called a Proposition 8 reduction. When the market recovers, your value is restored up to the Prop 13 factored base year value — never above it.
Source: BOE Publication 9 · P13-006

## P13-007 | Prop 13 Basics | Difference between assessed value and market value?
Market value is what your property would sell for today. Assessed value is what the county uses to calculate taxes. Because of the 2% annual cap, assessed value may be significantly lower than market value, especially for long-term owners.
Source: BOE Publication 29 · P13-007

## P13-008 | Prop 13 Basics | I just bought a home. How will taxes be calculated?
Your purchase price becomes your new base year value. Annual tax is approximately 1% of that value plus local charges. Assessed value can only increase by up to 2% per year going forward unless you make significant improvements or the property changes ownership again.
Source: BOE Publication 29 · P13-008

## CIO-001 | Change in Ownership | What is a Change in Ownership?
A Change in Ownership (CIO) is a transfer of property that causes the county to reassess it at current market value. The new market value becomes the new base year value, subject to Prop 13 rules going forward. This can significantly increase taxes if market value has risen.
Source: BOE Publication 150-C · CIO-001

## CIO-002 | Change in Ownership | What events trigger a Change in Ownership?
Common triggers: sale or purchase of property, a gift of real property, transferring into or out of certain trusts, transferring into or out of an LLC or corporation, and inheriting property in some circumstances. Any transfer of beneficial ownership — regardless of whether money changes hands — can trigger reassessment.
Source: BOE Publication 150-C · CIO-002

## CIO-003 | Change in Ownership | What does NOT trigger a Change in Ownership?
Exclusions include: transfers between spouses, transfers between registered domestic partners, certain parent-child transfers, certain grandparent-grandchild transfers, transfers into a revocable living trust where the transferor remains the beneficiary, and transfers that only change how title is held without changing actual ownership.
Source: BOE Publication 150-C · CIO-003

## CIO-004 | Change in Ownership | What is the Parent-Child Exclusion?
Under Proposition 19 (effective February 16, 2021), the exclusion applies only to a primary residence, and only if the child uses it as their primary residence within one year of the transfer. The prior exclusion for rental or vacation properties was largely eliminated. File form BOE-58-AH to claim.
Source: BOE Publication 150-C · CIO-004

## CIO-005 | Change in Ownership | What is the Grandparent-Grandchild Exclusion?
Under Prop 19: applies only to a primary residence, only if all of the grandchild's parents who are children of the grandparent are deceased, and only if the grandchild moves in within one year. File form BOE-58-AH.
Source: BOE Publication 150-C · CIO-005

## CIO-006 | Change in Ownership | Does transferring to my spouse trigger reassessment?
No. Transfers between spouses are excluded, including transfers during marriage, as a result of divorce, or upon death. Registered domestic partners receive the same exclusion.
Source: BOE Publication 150-C · CIO-006

## CIO-007 | Change in Ownership | I inherited a property. Will it be reassessed?
It depends. Inheriting a property can trigger reassessment, but exclusions may apply. Under Prop 19, inheriting a primary residence from a parent or grandparent and using it as your primary residence within one year may qualify for an exclusion. Rental and investment properties are generally reassessed.
Source: BOE Publication 150-C · CIO-007

## CIO-008 | Change in Ownership | Does transferring into a trust trigger reassessment?
Transferring into a revocable living trust where you remain the beneficiary does not trigger reassessment. Transferring into an irrevocable trust or one where someone else becomes the beneficial owner may trigger a CIO.
Source: BOE Publication 150-C · CIO-008

## CIO-010 | Change in Ownership | What is a Preliminary Change of Ownership Report (PCOR)?
A PCOR (form BOE-502-A) must be filed with the county recorder whenever real property is transferred. It helps the Assessor determine whether reassessment applies or an exclusion exists. Failure to file may result in a $20 fee.
Source: BOE Publication 4 · CIO-010

## NC-001 | New Construction | What counts as new construction for property tax purposes?
New construction includes any addition of a new structure, expansion of an existing structure, and any conversion requiring a building permit. Only the newly constructed portion is reassessed — the rest retains its existing base year value.
Source: BOE Publication 150-D · NC-001

## NC-002 | New Construction | Do repairs and maintenance trigger reassessment?
No. Routine repairs such as replacing a roof with the same materials, repainting, or repairing plumbing do not trigger reassessment. Reassessment requires that work adds value, extends useful life beyond original condition, or converts to a new use.
Source: BOE Publication 150-D · NC-002

## NC-004 | New Construction | I added a room. How will my taxes be affected?
Adding a room triggers reassessment but only for the new addition. The Assessor determines the market value of the added square footage as of completion date. Your existing home retains its current base year value.
Source: BOE Publication 150-D · NC-004

## NC-007 | New Construction | I built an ADU. Will my taxes go up?
Yes, in most cases. A newly constructed ADU is new construction and will be assessed at market value when substantially completed. Only the ADU is reassessed; your existing home and land retain their current assessed values.
Source: BOE Publication 150-D · NC-007

## NC-008 | New Construction | Are solar panels considered new construction?
Active solar energy systems are currently excluded from new construction reassessment under California law (R&T Code section 73). This exclusion expires January 1, 2027, unless extended by the legislature.
Source: BOE Publication 150-D · NC-008

## NC-010 | New Construction | Are accessibility improvements excluded?
Yes. Improvements made to accommodate a disabled person — including ramps, widened doorways, and modified bathrooms — are excluded from new construction reassessment.
Source: BOE Publication 150-D · NC-010

## SUP-001 | Supplemental Assessments | What is a supplemental assessment?
A supplemental assessment is issued when a property's taxable value increases mid-year due to a Change in Ownership or new construction. It captures the value increase between the event date and fiscal year end, resulting in a separate supplemental tax bill.
Source: BOE Publication 26 · SUP-001

## SUP-002 | Supplemental Assessments | Why did I get a supplemental bill after buying my home?
Your purchase price becomes your new base year value. If higher than the prior owner's assessed value, the county issues a supplemental assessment covering the portion of the fiscal year after your purchase. It is a one-time bill — not collected through most mortgage impound accounts.
Source: BOE Publication 26 · SUP-002

## SUP-003 | Supplemental Assessments | How is the supplemental assessment calculated?
The difference between new and prior assessed values is prorated by months remaining in the fiscal year (July 1 to June 30). An event in July results in a nearly full-year bill; an event in May results in a small one.
Source: BOE Publication 26 · SUP-003

## SUP-004 | Supplemental Assessments | Can I receive more than one supplemental bill?
Yes. If your event occurs between January 1 and May 31, you may receive two bills: one for the remainder of the current fiscal year and one for the following fiscal year.
Source: BOE Publication 26 · SUP-004

## SUP-006 | Supplemental Assessments | Is a supplemental bill the same as my regular tax bill?
No. Your regular annual bill covers the full fiscal year and is typically paid through your mortgage impound account. The supplemental bill is a separate, one-time bill that your lender usually does not pay. You are responsible for paying it directly.
Source: BOE Publication 26 · SUP-006

## SUP-007 | Supplemental Assessments | What if my supplemental assessment is too high?
You have the right to appeal. The deadline is 60 days from the date of the Notice of Supplemental Assessment — shorter than the regular roll appeal window. Contact the Assessment Appeals Board at (619) 531-5600.
Source: BOE Publication 26 · SUP-007

## EX-HO-001 | Homeowners Exemption | What is the Homeowners Exemption?
The Homeowners Exemption reduces your assessed value by $7,000, saving most homeowners roughly $70 per year. Available to California homeowners who own and occupy their home as their primary residence as of January 1. Renews automatically once filed.
Source: BOE Publication 2 · EX-HO-001

## EX-HO-002 | Homeowners Exemption | How do I apply for the Homeowners Exemption?
File form BOE-266 with the San Diego County Assessor's office. Filing by February 15 gives the full exemption. Filing between February 16 and December 10 gives 80% for that year. The exemption renews automatically.
Source: BOE Publication 2 · EX-HO-002

## EX-HO-003 | Homeowners Exemption | Who qualifies?
You must own the property and occupy it as your primary residence as of January 1. Applies to one property only. Renters, landlords, and owners of vacation or investment properties do not qualify.
Source: BOE Publication 2 · EX-HO-003

## EX-HO-004 | Homeowners Exemption | I just bought a home. Do I need to apply?
Yes. The exemption does not transfer automatically. File a new claim (BOE-266). The Assessor may mail you a form after your purchase records, but it is your responsibility to file.
Source: BOE Publication 2 · EX-HO-004

## EX-VET-DV-001 | Disabled Veterans Exemption | What is the Disabled Veterans Exemption?
A significant property tax benefit for veterans with a service-connected disability or their surviving spouses. Two tiers: base exemption (approx. $161,083 assessed value reduction) and low-income exemption (approx. $241,627). Both amounts are adjusted annually by the BOE.
Source: BOE Publication 149 · EX-VET-DV-001

## EX-VET-DV-002 | Disabled Veterans Exemption | Who qualifies?
Veterans honorably discharged with a 100% service-connected VA disability rating, or rated totally disabled. Also veterans who are blind in both eyes or have lost use of two or more limbs. Property must be the veteran's primary residence.
Source: BOE Publication 149 · EX-VET-DV-002

## EX-VET-DV-004 | Disabled Veterans Exemption | How do I apply?
File form BOE-261-G with the San Diego County Assessor's office, along with your DD-214 and VA disability rating letter. For the low-income tier, include income documentation. Must be renewed annually by February 15.
Source: BOE Publication 149 · EX-VET-DV-004

## EX-VET-DV-005 | Disabled Veterans Exemption | Does it need to be renewed every year?
Yes. Unlike the Homeowners Exemption, the Disabled Veterans Exemption requires annual renewal by February 15. The Assessor mails renewal forms in late fall, but filing is your responsibility.
Source: BOE Publication 149 · EX-VET-DV-005

## EX-VET-B-001 | Basic Veterans Exemption | What is the Basic Veterans Exemption?
Reduces assessed value by $4,000 (approx. $40/year savings). Available to veterans honorably discharged who served during a qualifying war period, and their unmarried surviving spouses. No disability rating required.
Source: BOE Publication 1 · EX-VET-B-001

## EX-VET-B-003 | Basic Veterans Exemption | How do I apply?
File form BOE-261 with the San Diego County Assessor's office along with your DD-214. File by February 15 for the full exemption. Renews automatically once granted.
Source: BOE Publication 1 · EX-VET-B-003

## BILL-001 | Billing and Payments | When are property tax bills mailed?
San Diego County property tax bills are mailed in late October, covering the full fiscal year July 1 through June 30.
Source: SD County Treasurer-Tax Collector · BILL-001

## BILL-002 | Billing and Payments | When are property taxes due?
Two installments: First installment due November 1, delinquent after December 10. Second installment due February 1, delinquent after April 10. Both can be paid upfront if preferred.
Source: SD County Treasurer-Tax Collector · BILL-002

## BILL-003 | Billing and Payments | What happens if I miss a deadline?
A 10% penalty is added. If the second installment is not paid by June 30, the property becomes tax-defaulted with an additional $33 redemption fee. After five years of non-payment, the property may be subject to the county tax sale process.
Source: SD County Treasurer-Tax Collector · BILL-003

## BILL-004 | Billing and Payments | How can I pay my property taxes?
Online at sdttc.com (free e-check, or credit/debit card with convenience fee), by phone, by mail, or in person at 1600 Pacific Highway, Room 162, San Diego, CA 92101. Call (877) 829-4732.
Source: SD County Treasurer-Tax Collector · BILL-004

## BILL-009 | Billing and Payments | What is Mello-Roos?
Mello-Roos is a special tax for properties within a Community Facilities District (CFD), funding infrastructure like schools and roads. It appears as a separate line item on your bill and is not based on assessed value.
Source: BOE Publication 29 · BILL-009

## BILL-010 | Billing and Payments | Can I get a penalty waived?
Only in very limited circumstances: documented medical emergency, natural disaster, or county error. Financial hardship alone is not sufficient. You must pay the full amount including penalty first, then submit a penalty cancellation request with documentation to the Treasurer-Tax Collector.
Source: SD County Treasurer-Tax Collector · BILL-010

## APP-001 | Assessment Appeals | What is an assessment appeal?
A formal process to challenge the assessed value assigned to your property. If you believe your assessed value exceeds market value, you can appeal to the Assessment Appeals Board — an independent body separate from the Assessor. A successful appeal can reduce your taxes.
Source: BOE Publication 30 · APP-001

## APP-002 | Assessment Appeals | What are the grounds for appeal?
Most common: assessed value exceeds fair market value as of the applicable date. You may also appeal improperly denied exemptions, incorrect new construction valuations, or supplemental assessment values. You cannot appeal the tax rate or Mello-Roos charges.
Source: BOE Publication 30 · APP-002

## APP-003 | Assessment Appeals | What are the filing deadlines?
Regular roll: July 2 through November 30 of the assessment year. Supplemental assessments: 60 days from the Notice of Supplemental Assessment. Escape assessments: 60 days from notice. Missing a deadline generally forfeits your right to appeal for that year.
Source: BOE Publication 30 · APP-003

## APP-004 | Assessment Appeals | How do I file an appeal?
Submit an Application for Changed Assessment to the San Diego County Clerk of the Board of Supervisors. There is a filing fee. You must continue paying taxes on time during the appeal to avoid penalties. Contact (619) 531-5600.
Source: SD County Clerk of the Board · APP-004

## APP-007 | Assessment Appeals | Do I keep paying taxes while my appeal is pending?
Yes. Filing an appeal does not suspend your payment obligation. If your appeal succeeds and the assessed value is reduced, you will receive a refund for any overpayment, with interest in some cases.
Source: BOE Publication 30 · APP-007

## P8-001 | Prop 8 Reductions | What is a Proposition 8 reduction?
If your property's current market value falls below its Prop 13 factored base year value, the county must temporarily reduce your assessed value to the lower market value. This is temporary — not a permanent change to your base year value.
Source: BOE Publication 9 · P8-001

## P8-004 | Prop 8 Reductions | Will my taxes go back up when the market recovers?
Yes. As the market recovers, your assessed value is restored up to the Prop 13 factored base year value — never above it. During recovery, your value can increase by more than 2% per year until it reaches the factored base year value.
Source: BOE Publication 9 · P8-004

## P8-005 | Prop 8 Reductions | Can assessed value increase by more than 2% after a Prop 8 reduction?
Yes. The standard 2% annual cap applies only when enrolled at the factored base year value. During recovery from a Prop 8 reduction, increases can exceed 2% per year until returning to the Prop 13 cap.
Source: BOE Publication 9 · P8-005
"""


# ---------------------------------------------------------------------------
# 2. CHUNK PARSER
#    Parses the raw KB string into structured chunk dicts.
# ---------------------------------------------------------------------------

def parse_knowledge_base(raw: str) -> list[dict]:
    chunks = []
    current = None
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("## "):
            if current:
                chunks.append(current)
            parts = line[3:].split(" | ", 2)
            current = {
                "id": parts[0] if len(parts) > 0 else "",
                "topic": parts[1] if len(parts) > 1 else "",
                "question": parts[2] if len(parts) > 2 else "",
                "content": "",
                "source": ""
            }
        elif line.startswith("Source:") and current:
            current["source"] = line[7:].strip()
        elif current is not None:
            current["content"] = (current["content"] + " " + line).strip()
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# 3. TF-IDF RETRIEVER
#    No external API needed. For production, swap with:
#    - OpenAI text-embedding-3-small  (~$0.00002 per query, excellent quality)
#    - sentence-transformers/all-MiniLM-L6-v2  (free, runs locally)
#    - Azure OpenAI embedding endpoint (if county has Azure contract)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r'[a-z]+', text.lower())

def build_tfidf_index(chunks: list[dict]) -> dict:
    """Build TF-IDF vectors for each chunk."""
    corpus = []
    for c in chunks:
        text = f"{c['topic']} {c['question']} {c['content']}"
        corpus.append(tokenize(text))

    # Document frequency
    df = Counter()
    for doc in corpus:
        for term in set(doc):
            df[term] += 1

    N = len(corpus)
    idf = {term: math.log(N / freq) for term, freq in df.items()}

    # TF-IDF vectors (sparse dicts)
    vectors = []
    for doc in corpus:
        tf = Counter(doc)
        total = len(doc)
        vec = {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}
        vectors.append(vec)

    return {"vectors": vectors, "idf": idf}

def cosine_similarity(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v**2 for v in a.values()))
    mag_b = math.sqrt(sum(v**2 for v in b.values()))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

def retrieve(query: str, chunks: list[dict], index: dict, top_k: int = 4) -> list[dict]:
    """Return the top_k most relevant chunks for a query."""
    idf = index["idf"]
    q_tokens = tokenize(query)
    tf = Counter(q_tokens)
    total = len(q_tokens)
    q_vec = {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}

    scores = [cosine_similarity(q_vec, v) for v in index["vectors"]]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked[:top_k] if scores[i] > 0]


# ---------------------------------------------------------------------------
# 4. PROMPT BUILDER
#    Only the retrieved chunks go to the model — not the full KB.
# ---------------------------------------------------------------------------

SYSTEM_BASE = """You are the San Diego County Property Tax Assistant. You help residents understand property taxes using only the provided reference material.

Rules:
- Answer only from the REFERENCE MATERIAL below. Do not use outside knowledge.
- Be clear and friendly. Define technical terms when you use them.
- End every answer with the Source line from the matching reference entry.
- If the question is outside the reference material, say so and direct the user to call the Assessor's office at (619) 236-3771.
- Never guess. If unsure, escalate to the office.
- Do not answer questions unrelated to San Diego County property taxes."""

def build_system_prompt(retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return SYSTEM_BASE + "\n\nNo relevant reference material found for this query."

    refs = "\n\n---\n\n".join(
        f"[{c['id']}] {c['topic']} — {c['question']}\n{c['content']}\nSource: {c['source']}"
        for c in retrieved_chunks
    )
    return f"{SYSTEM_BASE}\n\n=== REFERENCE MATERIAL ===\n\n{refs}"


# ---------------------------------------------------------------------------
# 5. CHAT LOOP
#    Maintains conversation history. Each turn:
#    - retrieves relevant chunks for the NEW user message
#    - rebuilds the system prompt with only those chunks
#    - calls Claude Haiku (cheapest capable model)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Very rough token estimate: ~4 chars per token."""
    return len(text) // 4

def chat():
    client = Anthropic()
    chunks = parse_knowledge_base(RAW_KNOWLEDGE_BASE)
    index = build_tfidf_index(chunks)
    history = []

    print(f"\nLoaded {len(chunks)} knowledge base chunks.")
    print("SD County Property Tax Assistant (RAG mode)")
    print("Model: claude-haiku-4-5  |  Type 'quit' to exit\n")
    print("-" * 60)

    session_input_tokens = 0
    session_output_tokens = 0

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        # Retrieve relevant chunks for this specific question
        retrieved = retrieve(user_input, chunks, index, top_k=4)
        system_prompt = build_system_prompt(retrieved)

        # Log what was retrieved (remove in production)
        chunk_ids = [c["id"] for c in retrieved]
        est_system_tokens = estimate_tokens(system_prompt)
        print(f"\n[RAG] Retrieved: {chunk_ids} | ~{est_system_tokens} system tokens")

        history.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=system_prompt,
            messages=history
        )

        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})

        # Token tracking
        session_input_tokens += response.usage.input_tokens
        session_output_tokens += response.usage.output_tokens

        # Haiku pricing: $0.80/M input, $4.00/M output (as of 2025)
        cost = (session_input_tokens * 0.80 + session_output_tokens * 4.00) / 1_000_000
        print(f"\nAssistant: {reply}")
        print(f"\n[Tokens this session: {session_input_tokens} in / {session_output_tokens} out | Est. cost: ${cost:.5f}]")

    print("\nSession ended.")


if __name__ == "__main__":
    chat()
