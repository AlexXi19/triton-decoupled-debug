# triton-decoupled-debug

Debugging issue: https://github.com/triton-inference-server/server/issues/5951

This repository attempts to debug issues with using the decoupled exec in a python bls with a triton server (ver. 23.04). The behavior is that the `res.send` hangs after a certain number of requests. 

Update: Jul 6

Unable to reproduce the error, moved to triton version 23.06 and haven't experienced the error 🤔
