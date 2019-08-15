#!/bin/bash


mpi_niagara 1 "python bin/make_cov.py v1.0.0_rc deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 -o" -t 80 --walltime "01:45:00"
mpi_niagara 1 "python bin/make_cov.py v1.0.0_rc boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 --o" -t 80 --walltime "02:00:00"
