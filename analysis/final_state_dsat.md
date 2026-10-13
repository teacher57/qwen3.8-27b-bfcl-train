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
