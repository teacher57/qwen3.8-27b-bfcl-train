# DSAT: multi-turn "wrong final state" failures (Qwen3.8-27B, BFCL v4)

**Scope:** the 199 multi-turn conversations BFCL marks `instance_state_mismatch`: after some turn, the simulated system
(files, car, tickets, tweets, bookings, orders, messages) differs from what the expected calls produce.
That's 63% of all 318 multi-turn failures.

**Method (no LLM involved):** `dsat_multi_turn.py` lines up the model's calls with the expected calls turn by turn and
finds the **first turn where the state-changing calls differ**. Calls that returned an error are ignored (they changed
nothing), and so are repeated identical calls and read-only calls (`ls`, `get_*`, login checks...).
`final_state_root_causes.py` then assigns root causes, named after reading samples of every cluster.
Per-conversation labels: `final_state_root_causes.json`. To browse them: `renders/bfcl_multi_turn.html`.

## Root causes

|       | Root cause                                                                           | All    | base | miss_func | miss_param | long_ctx |
| ----- | ------------------------------------------------------------------------------------ | ------ | ---- | --------- | ---------- | -------- |
| **A** | **Did extra steps nobody asked for**                                                 | **62** | 18   | 14        | 13         | 17       |
| A1    | released / re-pressed the brake after starting the engine                            | 39     | 10   | 10        | 7          | 12       |
| A3    | wrote invented content into a file it was only asked to create                       | 12     | 5    | 2         | 2          | 3        |
| A2    | started navigation when only asked to find a place                                   | 9      | 3    | 2         | 2          | 2        |
| A4    | set a budget limit nobody asked for                                                  | 2      | 0    | 0         | 2          | 0        |
| **B** | **Rewrote or embellished text the user gave** (tweet, ticket, message, file content) | **36** | 7    | 6         | 9          | 14       |
| **D** | **Guessed missing information instead of asking**                                    | **32** | –    | –         | 32         | –        |
| **C** | **Small exact-value mistakes** (format, unit, precision, wrong id/option)            | **25** | 9    | 3         | 6          | 7        |
| **E** | **Worked around a missing function instead of saying it can't**                      | **19** | –    | 19        | –          | –        |
| **F** | **Asked for confirmation/clarification when it should have acted**                   | **18** | 7    | 5         | 4          | 2        |
| G     | did nothing / just answered when action was expected                                 | 5      | 0    | 2         | 1          | 2        |
| H     | skipped a required action                                                            | 2      | 1    | 0         | 1          | 0        |

## What each one looks like

**A1, brake (39).** "Start the engine" → the model presses the brake, starts the engine, then **releases the brake**,
like a real driver would. None of the 39 users asked for that, and BFCL's expected state keeps the brake pressed.
Partly a benchmark convention ("touch nothing you weren't asked to"), but it's the model's single most common failure.

**A3, file content (12).** "Create a file `notes.md` to capture meeting highlights" → `touch` is enough, but the model
also `echo`s a made-up "# Meeting Highlights" template into it.

**A2, navigation (9).** "Find the nearest tire shop" → it finds it and also **sets the navigation** to it.

**B, rewriting the user's text (36).** The user gives exact text, or text to pass on, and the model "improves" it:
- `miss_param_6`: *"put 'Company Earning: 2000 Company Expenditure: 500 Company Name: Gorilla'"* → it reorders the
  three facts onto separate lines.
- `base_18`: *"post the differences [between two files]"* → instead of the diff, it posts a promotional tweet ("from
  ancient civilizations to modern world events, the journey continues!").
- Ticket and support messages: *"description 'Urgent assistance required.'"* → it adds a whole case summary.

**D, guessing (32).** The turn leaves out something needed (which file, how much fuel, which price) and the model fills
it in instead of asking. E.g. `miss_param_2`: *"draft up a document for keeping track of fresh ideas"* (no name, no
content) → it invents `TeamNotes.txt` and writes a full "Fresh Ideas Tracker" template.

**C, exact values (25).** `@` dropped from a mention (`roadsideassistance` vs `@RoadsideAssistance`), `.docx` added to a
name the user gave without one, the zip code dropped from an address, 4.5195 instead of 4.51947 ("four decimal places"
is the user's own wording; BFCL's expected value has five), wrong insurance type, the wrong access token.

**E, workarounds (19).** The needed function is withheld for one turn; instead of saying it can't do that yet, the
model reaches the goal through other functions.

**F, over-asking (18).** The opposite of D: everything needed is there, but the model stops to confirm. E.g.
`miss_param_122` (trading): *"Do you still want to place it at $150, or would you prefer a price closer to the
market?"* when the user had already given the order.

## The pattern behind it

The model's failures are mostly **not** about tool-calling skill (wrong functions or broken calls are rare: H = 2). They
are about **doing exactly what was asked, no more and no less**:

1. **Too much initiative (A + B: 98 of 199, half).** Extra steps and "improved" content: a helpful-assistant habit
   that BFCL, and real automation, treat as wrong.
2. **Wrong call on when to ask (D + E + F: 69).** Guessing when information is missing (D, E), yet asking when it isn't
   needed (F). Same skill both ways: knowing when information is missing.
3. **Exactness (C: 25).** Copying values exactly as given (prefixes, extensions, numbers).

## What would help (for the training discussion)

- **For 1:** preference data (DPO) where the "chosen" answer does exactly the request and the "rejected" one adds steps
  or rewrites the user's text. A system prompt rule ("do only what is asked; pass user-provided text through verbatim")
  is a cheap test first.
- **For 2:** conversations that teach **when** to ask: some with a missing parameter (ask), some complete (just act).
  This is the τ-bench habit too.
- **For 3:** small SFT sets for copying values verbatim.
- **Caveat:** some of A1 and C are BFCL conventions (keeping the brake pressed, five decimals) rather than clear
  mistakes. Don't overfit to them; check real-world-style behavior on τ³ at the same time.

## After SFT + GRPO: the same DSAT on the new eval (GRPO step 20)

The same 800 BFCL multi-turn entries, run again with the model after the fs_env SFT warm start and GRPO (step 20, the
best held-out checkpoint), same BFCL settings (temperature 0.001, thinking on, 16,384 tokens per call; context 131k
instead of 65k). The same two scripts, pointed at the new results (they reproduce the base-model labels above exactly):

```
BFCL_ROOT=results/fs_env/grpo/run_v1/bfcl BFCL_MODEL=qwen3.8-27b-grpo20-FC DSAT_OUT=analysis/grpo20 python3 analysis/dsat_multi_turn.py
BFCL_ROOT=results/fs_env/grpo/run_v1/bfcl BFCL_MODEL=qwen3.8-27b-grpo20-FC DSAT_OUT=analysis/grpo20 python3 analysis/final_state_root_causes.py
python3 analysis/compare_dsat.py      # prints the tables below
```

Labels per conversation: `analysis/grpo20/final_state_root_causes.json` and `analysis/grpo20/multi_turn_dsat.json`.

**All failed multi-turn entries, by first divergence** (base: 318, GRPO step 20: 295 of 800)

| What went wrong first | base | GRPO step 20 | change |
|---|---|---|---|
| Did an extra, unrequested action | 65 | 73 | +8 |
| Right action, wrong argument values | 72 | 64 | -8 |
| Guessed the missing information instead of asking | 44 | 34 | -10 |
| Skipped a lookup the answer needed | 23 | 33 | +10 |
| Worked around the missing function instead of saying it can't | 31 | 23 | -8 |
| Asked for confirmation/info instead of acting | 31 | 23 | -8 |
| No call-level divergence found (state differs for another reason) | 18 | 18 | 0 |
| Looped until the 20-step limit | 10 | 12 | +2 |
| Took no action (said it can't / just answered) | 11 | 6 | -5 |
| excluded: context limit (our setup) | 11 | 1 | -10 |
| Skipped a required action | 2 | 5 | +3 |
| Different approach (other functions than expected) | 0 | 3 | +3 |
| **total** | **318** | **295** | **-23** |

**Wrong-final-state failures by root cause** (base → GRPO step 20, per category)

| Root cause | all | base | miss_func | miss_param | long_context |
|---|---|---|---|---|---|
| A1 extra step: released / re-pressed the brake | 39 → **45** | 10 → 9 | 10 → 8 | 7 → 11 | 12 → 17 |
| A2 extra step: started navigation | 9 → **8** | 3 → 2 | 2 → 1 | 2 → 2 | 2 → 3 |
| A3 wrote invented content into a new file | 12 → **10** | 5 → 4 | 2 → 2 | 2 → 1 | 3 → 3 |
| A4 other unrequested action | 2 → **5** | 0 → 1 | 0 → 1 | 2 → 2 | 0 → 1 |
| B rewrote the user's exact text | 36 → **39** | 7 → 8 | 6 → 8 | 9 → 8 | 14 → 15 |
| C small exact-value mistake | 25 → **21** | 9 → 6 | 3 → 5 | 6 → 6 | 7 → 4 |
| D guessed missing info instead of asking | 32 → **27** | 0 → 0 | 0 → 0 | 32 → 27 | 0 → 0 |
| E worked around a missing function | 19 → **14** | 0 → 0 | 19 → 14 | 0 → 0 | 0 → 0 |
| F asked when it should act | 18 → **15** | 7 → 5 | 5 → 6 | 4 → 2 | 2 → 2 |
| G did nothing / just answered | 5 → **4** | 0 → 0 | 2 → 0 | 1 → 1 | 2 → 3 |
| H other (skipped action, different approach...) | 2 → **6** | 1 → 3 | 0 → 0 | 1 → 2 | 0 → 1 |
| **total** | 199 → **194** | 42 → 38 | 49 → 45 | 66 → 62 | 42 → 49 |

**Entry by entry** (the same 800 entries): what happened to each base wrong-final-state failure, and where GRPO step 20's came from

| Root cause | base failures | fixed (now pass) | same cause | other failure | GRPO step 20 failures | new (passed before) | from another failure |
|---|---|---|---|---|---|---|---|
| A1 extra step: released / re-pressed the brake | 39 | 12 | 23 | 4 | 45 | 17 | 5 |
| A2 extra step: started navigation | 9 | 1 | 6 | 2 | 8 | 0 | 2 |
| A3 wrote invented content into a new file | 12 | 2 | 6 | 4 | 10 | 0 | 4 |
| A4 other unrequested action | 2 | 0 | 2 | 0 | 5 | 1 | 2 |
| B rewrote the user's exact text | 36 | 5 | 31 | 0 | 39 | 3 | 5 |
| C small exact-value mistake | 25 | 8 | 13 | 4 | 21 | 3 | 5 |
| D guessed missing info instead of asking | 32 | 7 | 22 | 3 | 27 | 2 | 3 |
| E worked around a missing function | 19 | 7 | 11 | 1 | 14 | 1 | 2 |
| F asked when it should act | 18 | 6 | 9 | 3 | 15 | 4 | 2 |
| G did nothing / just answered | 5 | 1 | 2 | 2 | 4 | 0 | 2 |
| H other (skipped action, different approach...) | 2 | 0 | 2 | 0 | 6 | 2 | 2 |
| **total** | **199** | **49** | **127** | **23** | **194** | **33** | **34** |

**How to read it**

- **What we trained moved the right way, and cleanly.** E (worked around a missing function): 7 of 19 fixed, 1 new.
  D (guessed instead of asking): 7 of 32 fixed, 2 new. A3 (invented file content): 2 fixed, none new. In the first table,
  "guessed" falls 44 → 34 and "worked around" 31 → 23 across all failures.
- **The brake (A1) didn't get worse so much as it churned:** 12 of its 39 failures were fixed, but 17 entries that passed
  before now fail on it, mostly in long_context (12 → 17) and miss_param (7 → 11). Whether the model releases the brake
  seems to hinge on small differences in the conversation. A1 needs its own training signal; it won't move as a side effect.
- **Rewriting the user's text (B) is untouched:** 31 of 36 failures still fail the same way. Nothing in fs_env taught it
  (B was only a regression check).
- **One new pattern:** "skipped a lookup the answer needed" grew 23 → 33 among all failures (wrong answers rather than
  wrong states). The model may have learned to act with fewer looks first; worth watching if training goes further.
- **Context limit:** 11 → 1 failures, only because the new run had 131k context instead of 65k (not a model change).
- **Caveat:** 49 base wrong-state failures got fixed while 33 entries that passed before now end in a wrong state. With this much movement in both directions, single
  rows of 5–45 entries are noisy; trust the direction of the trained causes (D, E), not small differences elsewhere.
