#!/bin/bash
D=/root/.cache/huggingface/modules/transformers_modules/LocateAnything-3B
sed -n '93,131p' "$D/modeling_locateanything.py"