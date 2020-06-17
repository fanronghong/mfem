#!/bin/bash

make pnp_steadystate_protein
reset
exec &> >(tee "./out") # https://unix.stackexchange.com/questions/145651/using-exec-and-tee-to-redirect-logs-to-stdout-and-a-log-file-in-the-same-time

# ------>
./pnp_steadystate_protein -p 1 -lin gummel -dis cg
./pnp_steadystate_protein -p 1 -lin newton -dis cg
./pnp_steadystate_protein -p 1 -lin gummel -dis dg

