#!/bin/bash

make pnp_steadystate_box
reset

# https://unix.stackexchange.com/questions/145651/using-exec-and-tee-to-redirect-logs-to-stdout-and-a-log-file-in-the-same-time
exec &> >(tee "./out")


./pnp_steadystate_box -p 1 -ref 0 -lin gummel -des cg
./pnp_steadystate_box -p 2 -ref 0 -lin gummel -des cg
