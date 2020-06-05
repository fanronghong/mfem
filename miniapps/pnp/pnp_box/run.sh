#!/bin/bash

make -j8 pnp_steadystate_box
reset

# https://unix.stackexchange.com/questions/145651/using-exec-and-tee-to-redirect-logs-to-stdout-and-a-log-file-in-the-same-time
exec &> >(tee "./out")


#./pnp_steadystate_box -p 1 -ref 0 -lin gummel -dis cg
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


./pnp_steadystate_box -p 1 -ref 0 -lin newton -dis cg -opts newton_cg_box_petsc_opts
./pnp_steadystate_box -p 1 -ref 0 -lin newton -dis dg -opts newton_dg_box_petsc_opts
./pnp_steadystate_box -p 2 -ref 0 -lin newton -dis cg -opts newton_cg_box_petsc_opts
./pnp_steadystate_box -p 2 -ref 0 -lin newton -dis dg -opts newton_dg_box_petsc_opts
./pnp_steadystate_box -p 1 -ref 1 -lin newton -dis cg -opts newton_cg_box_petsc_opts
./pnp_steadystate_box -p 1 -ref 1 -lin newton -dis dg -opts newton_dg_box_petsc_opts
./pnp_steadystate_box -p 2 -ref 1 -lin newton -dis cg -opts newton_cg_box_petsc_opts
./pnp_steadystate_box -p 2 -ref 1 -lin newton -dis dg -opts newton_dg_box_petsc_opts

