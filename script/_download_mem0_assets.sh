#!/bin/bash

huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir policy/Mem-0/checkpoints/Qwen3-VL-2B-Instruct
huggingface-cli download qiuly/Mem-0-m1mix-RMBench --local-dir policy/Mem-0/checkpoints/Mem-0-m1mix-RMBench

cd policy/Mem-0/checkpoints/Mem-0-m1mix-RMBench/checkpoint/
cat m1_mix_final_step50000.pt.part?? > m1_mix_final_step50000.pt
sha256sum -c m1_mix_final_step50000.pt.sha256