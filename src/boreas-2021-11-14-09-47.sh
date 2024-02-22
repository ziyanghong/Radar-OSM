#!/bin/bash
DO_DISPLAYMAP=0

DO_FILTER=1
DO_EXTRACT=1
DO_COMPERR=1
DO_PLOTERRS=0
DO_DISPLAY=0
DO_PLOTRESULTS=0
# MPICMD="mpirun -n 12"
# MPICMD="mpirun -n 1"

SCALES="1.0"


CFGFILES="configs/radar/boreas-2021-11-14-09-47.dcfg"
SEQUENCE_NAME=$(basename "$0")
SEQUENCE_NAME=${SEQUENCE_NAME/.sh//}
VIS_FOLDER="results/visualization/${SEQUENCE_NAME}"
mkdir -p $VIS_FOLDER




SECONDS=0
if [ $DO_FILTER -ne 0 ]; then
  for cfg in $CFGFILES; do
    for scl in $SCALES; do
     ./map_localization.py  --mode=filter --start_frame=250 --out_dir=results/radar/ ${cfg} || exit 
    done
  done
fi

if [ $DO_EXTRACT -ne 0 ]; then
  for cfg in $CFGFILES; do
    for scl in $SCALES; do
     ./map_localization.py  --mode=extract_modes  --out_dir=results/radar/ ${cfg} || exit 1
    done
  done
fi
if [ $DO_COMPERR -ne 0 ]; then
 for scl in $SCALES; do
    ./map_localization.py --mode=convert_gps_data  --out_dir=results/radar/ $CFGFILES || exit 1
 done
 for scl in $SCALES; do
    ./map_localization.py --mode=compute_errors  --out_dir=results/radar/ $CFGFILES || exit 1
 done
fi

if [ $DO_DISPLAY -ne 0 ]; then
  for cfg in $CFGFILES; do
    for scl in $SCALES; do
     ./map_localization.py  --mode=display  --out_dir=results/radar/ ${cfg} || exit 1
    done
  done
fi

CFGFILES="configs/radar/boreas-2021-11-14-09-47.dcfg"
if [ $DO_PLOTRESULTS -ne 0 ]; then
   ./map_localization.py --mode=plot_trajectories --out_dir=results/radar/ $CFGFILES || exit 1

   # ./map_localization.py --mode=compute_errors_ziyang --out_dir=results/radar/ $CFGFILES || exit 1 
fi

if [ $DO_DISPLAYMAP -ne 0 ]; then

  ./map_localization.py  --mode=display_map  --map=/home/hong/Documents/RadarOSM/data/map_glen_shields.bin  --img_format=pdf  || exit 1

fi