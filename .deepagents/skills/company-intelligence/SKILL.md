---
name: company-intelligence
description: Perform domain-specific web search for company intelligence—funding, leadership, competitors, recent news, and market position. Use when researching companies, due diligence, or competitive analysis.
version: 1.0.0
tags: [research, companies, web-search, competitive-intelligence]
---

# Company Intelligence Web Search

Use this skill when the user needs **research on companies**: funding, executives, competitors, recent news, or market context. It guides how to query and structure results for reliability and clarity.

## When to Use

- User asks about a company’s funding, investors, or valuation
- User wants to know leadership, key executives, or board members
- User asks about competitors or market position
- User wants recent news, product launches, or partnerships
- Any “research company X” or “tell me about company Y” style request

## Query Best Practices

1. **Be specific**  
   Prefer `"Acme Corp Series B funding 2025"` over `"Acme Corp funding"`.

2. **Use domain-focused phrases**  
   - Funding: `"[Company] funding round"`, `"[Company] investors"`
   - Leadership: `"[Company] CEO"`, `"[Company] leadership team"`
   - Competitors: `"[Company] competitors"`, `"[Company] vs [competitor]"`
   - News: `"[Company] news 2025"`, `"[Company] acquisition"` or `"partnership"`

3. **Run multiple queries**  
   Use 2–4 focused queries instead of one broad query (e.g. one for funding, one for leadership, one for recent news).

4. **Prefer recent and official**  
   When relevant, add time bounds (e.g. “2025”) and prefer official company or reputable press sources.

## Output Format

Structure the answer so it’s easy to act on:

```markdown
## [Company Name] – Summary
1–2 sentence overview (what they do, stage, geography if relevant).

## Key Facts
- **Funding:** [round, amount, date, lead investors if known]
- **Leadership:** [CEO and 1–2 other key roles]
- **Competitors / market:** [2–3 names or brief context]
- **Recent news:** [1–2 most relevant items with date]

## Sources
- [Title](URL) – [1 line note]
- ...
```

Keep the main reply under 400 words; add “Sources” with URLs so users can verify.

## Quality Checks

- **Cite sources** for funding amounts, executive names, and news.
- **Note dates** (e.g. “as of Q2 2025”) when information is time-sensitive.
- If results are thin or conflicting, say so and suggest a more specific or alternative query.
