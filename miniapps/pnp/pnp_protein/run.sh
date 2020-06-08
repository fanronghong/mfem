#!/bin/bash

reset

# https://unix.stackexchange.com/questions/145651/using-exec-and-tee-to-redirect-logs-to-stdout-and-a-log-file-in-the-same-time
exec &> >(tee "./out")

# ------>
./pnp_steadystate_protein -p 1 -lin gummel -dis cg -v
#./pnp_steadystate_protein -p 1 -lin newton -dis cg


