#!/bin/bash
# waits for run.sh to finish, then repairs the 4 no-tool live_irrelevance entries (fix_empty_tools.sh)
until grep -q RUN_DONE /root/pipeline.log; do sleep 60; done
/root/fix_empty_tools.sh >> /root/fix_empty_tools.out 2>&1 || echo "$(date -u +%FT%TZ) fix FAILED (see /root/fix_empty_tools.out)" >> /root/pipeline.log
