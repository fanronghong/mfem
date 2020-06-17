#!/bin/bash

make pnp_steadystate_box
reset
exec &> >(tee "./out") # https://unix.stackexchange.com/questions/145651/using-exec-and-tee-to-redirect-logs-to-stdout-and-a-log-file-in-the-same-time

# ------> 用解析解计算收敛阶
#./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis cg -rate
#./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis dg -rate
./pnp_steadystate_box -p 1 -ref 2 -lin newton -dis cg -rate
#./pnp_steadystate_box -p 1 -ref 2 -lin newton -dis dg -rate

# ------> Some Errors
# np1 not converge at 1st Gummel, leading to wrong convergence rate
#./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis cg
#./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis dg

