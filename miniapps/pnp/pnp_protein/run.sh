#!/bin/bash

reset

# https://unix.stackexchange.com/questions/145651/using-exec-and-tee-to-redirect-logs-to-stdout-and-a-log-file-in-the-same-time
exec &> >(tee "./out")

# ------> 用解析解计算收敛阶
#./pnp_steadystate_protein -p 1 -ref 2 -lin gummel -dis cg
./pnp_steadystate_protein -p 1 -ref 2 -lin newton -dis cg


