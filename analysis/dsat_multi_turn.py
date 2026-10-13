"""DSAT for BFCL multi-turn failures (Qwen3.8-27B): for every failed conversation, line up the model's calls with the
ground-truth calls turn by turn, find the FIRST turn where they diverge, and label how. No LLM involved.
    python3 analysis/dsat_multi_turn.py   ->  analysis/multi_turn_dsat.json (one record per failure) + printed summary
Cluster names are mechanical; analysis/multi_turn_dsat.md has the root causes after reading samples of each cluster."""
import ast, collections, json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results/pod/bfcl-results/result/qwen3.8-27b-FC/multi_turn"
SCO = HERE / "results/pod/bfcl-results/score/qwen3.8-27b-FC/multi_turn"
CATS = ["base", "miss_func", "miss_param", "long_context"]

# calls that only look at things (or move around / log in) and don't change what BFCL compares
READ = {"cat", "check_tire_pressure", "compute_exchange_rate", "diff", "displayCarStatus", "du", "estimate_distance",
        "estimate_drive_feasibility_by_mileage", "find", "find_nearest_tire_shop", "gallon_to_liter", "grep",
        "list_all_airports", "liter_to_gallon", "logarithm", "ls", "mean", "sort", "standard_deviation", "tail",
        "view_messages_sent", "wc", "retrieve_invoice", "verify_traveler_information", "pwd", "list_users",
        "absolute_value", "add", "divide", "multiply", "subtract", "power", "square_root", "round_number", "percentage",
        "max_value", "min_value", "sum_values", "si_unit_conversion", "imperial_si_conversion", "get_symbol_by_name",
        "filter_available_stocks_by_market_cap", "get_transaction_history", "list_messages", "search_messages",
        "get_message_stats", "get_tweet", "get_user_tweets", "get_tweet_comments", "get_user_stats", "list_all_users",
        "get_zipcode_based_on_city", "get_nearest_airport_by_city", "get_flight_cost", "get_credit_card_balance",
        "get_all_credit_cards", "get_budget_fiscal_year", "get_booking_history", "get_ticket", "get_user_tickets",
        "get_outside_temperature_from_google", "get_outside_temperature_from_weather_com", "get_current_speed",
        "get_zipcode_based_on_city", "update_market_status", "get_current_time", "get_symbol_by_name"}
NAV = {"cd", "authenticate_travel", "authenticate_twitter", "message_login", "ticket_login", "login_device",
       "trading_login", "trading_logout", "message_get_login_status", "ticket_get_login_status", "get_user_id"}


def is_read(name):
    return (name in READ or name in NAV or name.endswith("_get_login_status")
            or name.startswith(("get_", "list_", "display", "check_", "view_", "estimate_", "search_")))


def parse_expected(call):
    node = ast.parse(call, mode="eval").body
    args = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    return node.func.id, args


def norm(v):
    if isinstance(v, str): return v.strip().lower()
    if isinstance(v, float) and v.is_integer(): return int(v)
    if isinstance(v, list): return [norm(x) for x in v]
    if isinstance(v, dict): return {k: norm(x) for k, x in v.items()}
    return v


def key(name, args):
    return (name, json.dumps(norm(args), sort_keys=True, ensure_ascii=False))


def model_turns(result):
    """per turn: list of (name, args) the model called, final text, n_steps, and whether a user-facing question was asked"""
    out = []
    for block in (result.get("inference_log") or [])[1:]:
        if not isinstance(block, dict): continue
        calls, texts, steps, failed = [], [], 0, 0
        for k, msgs in block.items():
            if not k.startswith("step_"): continue
            steps += 1
            step_calls, results = [], []
            for m in msgs:
                c = m.get("content")
                if m.get("role") == "assistant":
                    if isinstance(c, list):
                        for d in c:
                            if isinstance(d, dict):
                                for n, a in d.items():
                                    try: a = json.loads(a) if isinstance(a, str) else a
                                    except ValueError: a = {"_raw": a}
                                    step_calls.append((n, a if isinstance(a, dict) else {"_raw": a}))
                    elif isinstance(c, str) and c.strip():
                        texts.append(c.strip())
                elif m.get("role") == "tool":
                    results.append(str(c))
            # tool results come back in call order; a call whose result is an error changed nothing, so drop it
            for i, call in enumerate(step_calls):
                res = results[i] if i < len(results) else ""
                if '"error"' in res[:200].lower() or res.lower().startswith("error"):
                    failed += 1; continue
                calls.append(call)
        out.append({"calls": calls, "text": "\n".join(texts), "steps": steps, "failed_calls": failed})
    return out


def classify(cat, exp_turns, mturns, err_type):
    if err_type == "multi_turn:inference_error":
        return None, "excluded: context limit (our setup)", {}
    if err_type == "multi_turn:force_terminated":
        # find the turn that ran out of steps
        t = next((i for i, m in enumerate(mturns) if m["steps"] >= 20), len(mturns) - 1)
        names = collections.Counter(n for n, _ in mturns[t]["calls"]) if mturns else {}
        return t, "Looped until the 20-step limit", {"most_called": names.most_common(3)}
    for t, exp in enumerate(exp_turns):
        m = mturns[t] if t < len(mturns) else {"calls": [], "text": "", "steps": 0}
        e_all = [parse_expected(c) for c in exp]
        e_w = collections.Counter(set(key(n, a) for n, a in e_all if not is_read(n)))
        m_w = collections.Counter(set(key(n, a) for n, a in m["calls"] if not is_read(n)))
        e_r = collections.Counter(key(n, a) for n, a in e_all if is_read(n) and n not in NAV)
        m_r = collections.Counter(key(n, a) for n, a in m["calls"] if is_read(n) and n not in NAV)
        info = {"failed_calls_in_turn": m.get("failed_calls", 0), "expected": exp, "model_calls": [f"{n}({json.dumps(a, ensure_ascii=False)[:160]})" for n, a in m["calls"]][:25],
                "model_text": m["text"][-500:]}
        if not exp:   # the turn where a parameter or function is missing: the right move is to ask / say it can't
            if m_w:
                lbl = {"miss_param": "Guessed the missing information instead of asking",
                       "miss_func": "Worked around the missing function instead of saying it can't"}.get(cat, "Acted in a turn where no action was expected")
                return t, lbl, info
            continue
        if e_w == m_w:
            if err_type == "multi_turn:execution_response_mismatch" and (e_r - m_r):
                info["missing_reads"] = list((e_r - m_r).elements())[:5]
                return t, "Skipped a lookup the answer needed", info
            continue
        e_names = collections.Counter(k[0] for k in e_w.elements()); m_names = collections.Counter(k[0] for k in m_w.elements())
        if not m_w:
            asked = "?" in m["text"][-400:]
            return t, "Asked for confirmation/info instead of acting" if asked else "Took no action (said it can't / just answered)", info
        if e_names == m_names:
            info["arg_diff"] = {"expected": [k for k in (e_w - m_w).elements()][:4], "model": [k for k in (m_w - e_w).elements()][:4]}
            return t, "Right action, wrong argument values", info
        missing, extra = e_names - m_names, m_names - e_names
        info["missing_actions"], info["extra_actions"] = list(missing.elements()), list(extra.elements())
        if missing and not extra: return t, "Skipped a required action", info
        if extra and not missing: return t, "Did an extra, unrequested action", info
        return t, "Different approach (other functions than expected)", info
    return None, "No call-level divergence found (state differs for another reason)", {}


records = []
for cat in CATS:
    results = {json.loads(l)["id"]: json.loads(l) for l in open(RES / f"BFCL_v4_multi_turn_{cat}_result.json") if l.strip()}
    fails = [json.loads(l) for l in list(open(SCO / f"BFCL_v4_multi_turn_{cat}_score.json"))[1:] if l.strip()]
    for f in fails:
        err = f.get("error") or {}
        mt = model_turns(results[f["id"]])
        t, label, info = classify(cat, f.get("possible_answer") or [], mt, err.get("error_type"))
        user = (f.get("prompt") or {}).get("question") or []
        records.append({"category": cat, "id": f["id"], "bfcl_error": err.get("error_type", "").split(":")[-1],
                        "cluster": label, "turn": None if t is None else t + 1, "n_turns": len(f.get("possible_answer") or []),
                        "user_message": (user[t][-1]["content"] if t is not None and t < len(user) and user[t] else "")[:400],
                        "classes": (f.get("prompt") or {}).get("involved_classes"), **info})

Path(HERE / "analysis").mkdir(exist_ok=True)
(HERE / "analysis/multi_turn_dsat.json").write_text(json.dumps(records, ensure_ascii=False, indent=1))
tab = collections.Counter((r["cluster"], r["category"]) for r in records)
clusters = collections.Counter(r["cluster"] for r in records)
print(f"{'cluster':58s} {'all':>4s} " + " ".join(f"{c[:10]:>10s}" for c in CATS))
for cl, n in clusters.most_common():
    print(f"{cl:58s} {n:4d} " + " ".join(f"{tab[(cl, c)]:10d}" for c in CATS))
print(f"{'total':58s} {len(records):4d}")
