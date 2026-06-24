#!/bin/bash

huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir policy/Mem-0/checkpoints/Qwen3-VL-2B-Instruct
huggingface-cli download qiuly/Mem-0-m1mix-RMBench --local-dir policy/Mem-0/checkpoints/Mem-0-m1mix-RMBench
