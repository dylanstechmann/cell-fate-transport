# Agent instructions — cell-fate-transport

Work only in this repository. This is balanced entropy-regularized optimal
transport between snapshots, plus a synthetic branch that is supposed to be
unidentifiable at the first time point (branch-B Brier worse than 0.25).

## Do not

- Switch the default to unbalanced / growth-aware transport.
- Infer proliferation, death, or reprogramming yield from changing sample counts.
- Treat UMAP distance as the cost.
- Delete or “fix” the unidentifiable-branch fixture so the demo looks smarter.
- Claim this is Waddington-OT. Cite Schiebinger / WOT as prior art, not as this code.

## First commands

```bash
python -m pip install -e ".[plots]"
python -m unittest discover -s tests -v
cellfate demo --out artifacts/agent-demo --plot
```

## Improve, in this order

1. Read `docs/METHODS.md`. Confirm marginal error and the Brier fixture still match the README numbers after your edit.
2. Allowed extension: a clearly named optional growth model **behind a flag that defaults off**, with a synthetic test where balanced OT is wrong and the flag is right. Document the flag in `REPORT.md` of the demo. Do not change default `cellfate infer` behavior.
3. If you touch couplings, keep ID alignment and `allow_pickle=False` loads.
4. Do not download a single-cell atlas into git.

## Done when

The bad-branch fixture still fails in the documented direction, and tests pass.

## Agent tooling (portable across harnesses)

<!-- agent-toolbox:installed -->

This repo vendors its agent tooling (no secrets inside) so Claude Code,
Cline, Codex, OpenCode, Kilo Code, Kimi Code and Grok share one setup:

- `.mcp.json` — project MCP servers `scholar` (PubMed, OpenAlex, Semantic
  Scholar, CORE, ORCID, Crossref) and `web` (Exa, Brave, Jina, Firecrawl).
- `opencode.json` — same servers for OpenCode; `AGENT_TOOLS.md` has the
  merge snippets for Codex / Cline / Kilo Code and notes for Kimi/Grok.
- `.claude/skills/` — project skills: citation-check, literature-search, web-research, secrets-guard, regen-guardrails, fold-providers.
- Keys: `~/.config/ai-keys.env` via the `agentkeys` loader — never in git,
  MCP config or chat. Run `secrets-guard` (`--staged`) before committing.
