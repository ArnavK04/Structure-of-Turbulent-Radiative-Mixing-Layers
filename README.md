This repository contains the scripts and modifications used for this project.

The simulations were performed using AthenaK (main branch, commit cc65bf80d8d52f2034b6b194ab959a6ef4a732b5 - March 6, 2025) from the official AthenaK repository (https://github.com/IAS-Astrophysics/athenak).

Only the files modified/added for this project are included here. The directory structure mirrors the AthenaK source tree. For example, a file located in vis/python/ in this repository should be placed in the corresponding athenak/vis/python/ directory inside the AthenaK source directory.

A short description of all folders

src - contains the problem generator file inside pgen/ and ism_cooling.hpp((has the modified cooling functions) inside srcterms/. outputs/history.cpp is a modified file with added diagnostics relevant to the TRML problem.

inputs - contains the input file 

vis/python - contains dignostic files

shell_scripts - contains shell scripts I uploaded to the repo as a backup.
