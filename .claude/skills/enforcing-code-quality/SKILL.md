---
name: enforcing-code-quality
description: Use when reviewing, refactoring, or cleaning up code - enforces systematic process (understand before changing), prevents regressions, requires verification before completion
---

# Enforcing Code Quality

## Overview

This skill enforces **process discipline** for code changes. Quality standards are in the code-quality-enforcer agent. This skill ensures you don't break things while improving them.

## The Iron Rule

**Never change code you don't understand. Never claim completion without verification.**

## The Process (Mandatory)

1. **UNDERSTAND**: Read the code. Trace dependencies. Find ALL usages with grep/search.
2. **ASK**: Clarify scope. Don't expand beyond what was requested.
3. **PLAN**: List specific changes. Identify what could break.
4. **CHANGE**: Make minimal, focused changes. One thing at a time.
5. **VERIFY**: Run tests. Check usages still work. Confirm no regressions.

## Red Flags - STOP if you think:

| Rationalization | Reality |
|-----------------|---------|
| "This is obviously unused" | Verify with grep first. Always. |
| "Quick fix, no need to trace" | Always trace dependencies. |
| "Tests passing = done" | Also verify runtime behavior. |
| "I'll clean this up while I'm here" | Stick to requested scope. |
| "This should also be fixed" | Ask before expanding scope. |
| "They said quick, so skip verification" | Time pressure doesn't justify skipping process. |

## Scope Control

**Before making changes, confirm:**
- What exactly was requested?
- Am I staying within that scope?
- If I found related issues, did I ASK before expanding?

**Never expand scope without explicit approval.**

## Project Rules (from CLAUDE.md)

- DO NOT implement features not explicitly requested
- DO NOT add "helpful" error handling without approval
- DO NOT change architecture without consent
- Stick to exactly what was asked - nothing more

## Verification Checklist

Before claiming completion:
- [ ] Searched for ALL usages of changed code
- [ ] Ran relevant tests
- [ ] Changes match requested scope (nothing extra)
- [ ] No new warnings or errors introduced
- [ ] Asked about anything unclear

## Common Mistakes

1. **Removing "unused" code without verifying** - Always grep the entire codebase
2. **Removing commented code** - Ask why it's there first, might be intentional
3. **Expanding scope** - "Found this too" requires asking permission
4. **Skipping tests** - Time pressure is not an excuse
5. **Making assumptions** - When in doubt, ask
