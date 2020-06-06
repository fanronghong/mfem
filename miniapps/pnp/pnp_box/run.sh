#!/bin/bash

reset

# https://unix.stackexchange.com/questions/145651/using-exec-and-tee-to-redirect-logs-to-stdout-and-a-log-file-in-the-same-time
exec &> >(tee "./out")

# 用gummel和cg计算收敛阶
#./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis cg -rate

# 用gummel和dg计算收敛阶
#./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis dg -rate

# 用gummel和cg计算收敛阶
#./pnp_steadystate_box -p 1 -ref 2 -lin newton -dis cg -rate

# 用gummel和dg计算收敛阶
#./pnp_steadystate_box -p 1 -ref 2 -lin newton -dis dg -rate

# ------> Some Errors
# np1 not converge at 1st Gummel, leading to wrong convergence rate
#./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis cg
./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis dg

#./pnp_steadystate_box -p 1 -ref 0 -lin gummel -dis dg
#./pnp_steadystate_box -p 1 -ref 1 -lin gummel -dis cg
#./pnp_steadystate_box -p 1 -ref 1 -lin gummel -dis dg
#./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis cg
#./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis dg
#./pnp_steadystate_box -p 1 -ref 3 -lin gummel -dis cg
#./pnp_steadystate_box -p 1 -ref 3 -lin gummel -dis dg

#./pnp_steadystate_box -p 1 -ref 0 -lin gummel -dis cg
#./pnp_steadystate_box -p 1 -ref 0 -lin gummel -dis dg
#./pnp_steadystate_box -p 2 -ref 0 -lin gummel -dis cg
#./pnp_steadystate_box -p 2 -ref 0 -lin gummel -dis dg
#./pnp_steadystate_box -p 1 -ref 1 -lin gummel -dis cg
#./pnp_steadystate_box -p 1 -ref 1 -lin gummel -dis dg
#./pnp_steadystate_box -p 2 -ref 1 -lin gummel -dis cg
#./pnp_steadystate_box -p 2 -ref 1 -lin gummel -dis dg
#./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis cg
#./pnp_steadystate_box -p 1 -ref 2 -lin gummel -dis dg
#./pnp_steadystate_box -p 2 -ref 2 -lin gummel -dis cg
#./pnp_steadystate_box -p 2 -ref 2 -lin gummel -dis dg


#./pnp_steadystate_box -p 1 -ref 0 -lin newton -dis cg
#./pnp_steadystate_box -p 1 -ref 0 -lin newton -dis dg
#./pnp_steadystate_box -p 1 -ref 1 -lin newton -dis cg
#./pnp_steadystate_box -p 1 -ref 1 -lin newton -dis dg
#./pnp_steadystate_box -p 1 -ref 2 -lin newton -dis cg
#./pnp_steadystate_box -p 1 -ref 2 -lin newton -dis dg


#./pnp_steadystate_box -p 1 -ref 0 -lin newton -dis cg
#./pnp_steadystate_box -p 1 -ref 0 -lin newton -dis dg
#./pnp_steadystate_box -p 2 -ref 0 -lin newton -dis cg
#./pnp_steadystate_box -p 2 -ref 0 -lin newton -dis dg
#./pnp_steadystate_box -p 1 -ref 1 -lin newton -dis cg
#./pnp_steadystate_box -p 1 -ref 1 -lin newton -dis dg
#./pnp_steadystate_box -p 2 -ref 1 -lin newton -dis cg
#./pnp_steadystate_box -p 2 -ref 1 -lin newton -dis dg


#./pnp_steadystate_box -p 1 -ref 2 -lin newton -dis cg -rate
#./pnp_steadystate_box -p 1 -ref 2 -lin newton -dis dg -rate

