# Invalid: the first A1-V results (2026-10-13)

`grpo-s20_original.jsonl` is the first, complete run of all 60 tasks. Its **A1-V** conversations (unzip withheld) are invalid:
the file system had a loophole. An archive copied to a name without `.zip` could be read as plain text, so the model could
get a file out without `unzip` (it did, in 38 of 40 conversations). The task demands "say you can't", which isn't fair when
it *is* possible. Fixed in `fs_env/filesystem.py` (a file is an archive by its content, not only its name), and A1-V was
re-run with the same seed. The other five scenarios didn't depend on it and are kept from this run.
