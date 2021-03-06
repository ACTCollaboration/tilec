#!/bin/bash

# TEST
#mpi_niagara 1 "python bin/actsim.py test_sim0 deep56 d56_01,d56_02 CMB 1.6 --overwrite  --nsims 1" -t 80 --walltime "00:15:00"

# Theory deep56 run
# mpi_niagara 4 "python bin/actsim.py test_sim_theory deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite  --nsims 8 --theory all" -t 40 --walltime "01:30:00"

# Baseline deep56 run with fgfit_v2 !!!!
# mpi_niagara 8 "python bin/actsim.py v1.2.0_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 " -t 40 --walltime "10:00:00"
# mpi_niagara 8 "python bin/actsim.py v1.2.0_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 80  " -t 40 --walltime "10:00:00"
# mpi_niagara 8 "python bin/actsim.py v1.2.0_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 160" -t 40 --walltime "10:00:00"
# mpi_niagara 8 "python bin/actsim.py v1.2.0_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 240  " -t 40 --walltime "10:00:00"
# mpi_niagara 8 "python bin/actsim.py v1.2.0_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 320" -t 40 --walltime "10:00:00"
# mpi_niagara 8 "python bin/actsim.py v1.2.0_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 400  " -t 40 --walltime "10:00:00"
# mpi_niagara 8 "python bin/actsim.py v1.2.0_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 480" -t 40 --walltime "10:00:00"
# mpi_niagara 8 "python bin/actsim.py v1.2.0_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 560  " -t 40 --walltime "10:00:00"


#python bin/actsim.py debug_v1.2.0_sim_baseline deep56 d56_05,d56_06 CMB 1.6 7.0 --overwrite --nsims 1 --start-index 273 --skip-inpainting --use-cached-sims

# Baseline deep56 run !!!!
# 0 - 79
#mpi_niagara 8 "python bin/actsim.py test_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 80" -t 40 --walltime "10:00:00"

# 80 - 159
#mpi_niagara 8 "python bin/actsim.py test_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 80 --start-index 80" -t 40 --walltime "10:00:00"



# Baseline deep56 run !!!!
# 0 - 79
#mpi_niagara 8 "python bin/actsim.py test_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 80" -t 40 --walltime "10:00:00"

# 80 - 159
#mpi_niagara 8 "python bin/actsim.py test_sim_baseline deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 80 --start-index 80" -t 40 --walltime "10:00:00"

#mpi_niagara 8 "python bin/actsim.py test_sim_baseline deep8 deep8_01,deep8_02,deep8_03,deep8_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 8" -t 40 --walltime "01:00:00"

#mpi_niagara 8 "python bin/actsim.py test_sim_baseline_theory deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 8 --theory all" -t 40 --walltime "01:00:00"


#mpi_niagara 4 "python bin/actsim.py test_sim_debug deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB 1.6 --overwrite --nsims 4 --skip-inpainting" -t 40 --walltime "01:00:00"

#mpi_niagara 4 "python bin/actsim.py test_sim_debug_fft deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB 1.6 --overwrite --nsims 4 --skip-inpainting --fft-beam" -t 40 --walltime "00:40:00"

#mpi_niagara 4 "python bin/actsim.py test_sim_debug_theory deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB 1.6 --overwrite --nsims 4 --skip-inpainting --theory all" -t 40 --walltime "00:40:00"

#mpi_niagara 8 "python bin/actsim.py test_sim_v2 deep56 p01,p02,p03,p04,p05,p06,p07,p08 CMB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 8 --skip-inpainting" -t 40 --walltime "00:30:00"

#mpi_niagara 8 "python bin/actsim.py test_sim_v2 deep56 p01,p02,p03,p04,p05,p06,p07,p08 CMB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 8 --skip-inpainting --theory all" -t 40 --walltime "00:30:00"

#mpi_niagara 4 "python bin/actsim.py test_lfi deep56 p01,p05 CMB 1.6 --overwrite --nsims 4 --skip-inpainting --theory all" -t 40 --walltime "00:30:00"


#mpi_niagara 8 "python bin/actsim.py test_sim_thsig deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 8" -t 40 --walltime "01:00:00"

#mpi_niagara 8 "python bin/actsim.py test_sim_nohigh deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 8" -t 40 --walltime "01:00:00"


# deep56 with more noise smoothing
#mpi_niagara 4 "python bin/actsim.py test_sim_delta_ell_800 deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite  --nsims 8 --delta-ell 800" -t 40 --walltime "01:30:00"

# deep56 with more signal smoothing
#mpi_niagara 4 "python bin/actsim.py test_sim_sigwidth_160 deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite  --nsims 8 --signal-bin-width 160" -t 40 --walltime "01:30:00"

# Baseline boss run with fgfit
# 0 - 79
mpi_niagara 16 "python bin/actsim.py v1.2.0_sim_baseline boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 " -t 80 --walltime "15:00:00"
mpi_niagara 16 "python bin/actsim.py v1.2.0_sim_baseline boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 80 " -t 80 --walltime "15:00:00"
mpi_niagara 16 "python bin/actsim.py v1.2.0_sim_baseline boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 160" -t 80 --walltime "15:00:00"
mpi_niagara 16 "python bin/actsim.py v1.2.0_sim_baseline boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 240 " -t 80 --walltime "15:00:00"
mpi_niagara 16 "python bin/actsim.py v1.2.0_sim_baseline boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 320" -t 80 --walltime "15:00:00"
mpi_niagara 16 "python bin/actsim.py v1.2.0_sim_baseline boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 400 " -t 80 --walltime "15:00:00"
mpi_niagara 16 "python bin/actsim.py v1.2.0_sim_baseline boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 480" -t 80 --walltime "15:00:00"
# mpi_niagara 16 "python bin/actsim.py v1.2.0_sim_baseline boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 7.0,7.0,10.0,10.0,10.0,10.0 --overwrite --nsims 80 --start-index 560 " -t 80 --walltime "15:00:00"


# Baseline boss run
# 0 - 79
#mpi_niagara 16 "python bin/actsim.py test_sim_baseline boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 80" -t 80 --walltime "15:00:00"

# 80 - 159
#mpi_niagara 16 "python bin/actsim.py test_sim_baseline boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 80 --start-index 80" -t 80 --walltime "15:00:00"

# boss theory
# mpi_niagara 8 "python bin/actsim.py test_sim_baseline_theory boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4 --overwrite --nsims 8 --theory all" -t 80 --walltime "02:30:00"


