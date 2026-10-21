#!/bin/bash
# N（519 位）进度查询
set -u
R=/root/rivermind-data
bash "$R/_k_fill_poll.sh" "$R/output/k72b_direct_519.json" "$R/k72b_n.log"
