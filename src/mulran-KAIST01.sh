#!/bin/bash

DO_FILTER=1
DO_EXTRACT=1
DO_COMPERR=1
DO_PLOTERRS=0
DO_DISPLAY=0
DO_PLOTRESULTS=0
# MPICMD="mpirun -n 12"
# MPICMD="mpirun -n 1"


SCALES="1.0"


CFGFILES="configs/radar/mulran-KAIST01.dcfg"

PLOT_PARAMS="--img_format=pdf"
SEQUENCE_NAME=$(basename "$0")
SEQUENCE_NAME=${SEQUENCE_NAME/.sh//}
VIS_FOLDER="results/visualization/${SEQUENCE_NAME}"
mkdir -p $VIS_FOLDER

SECONDS=0
if [ $DO_FILTER -ne 0 ]; then
  for cfg in $CFGFILES; do
    for scl in $SCALES; do
     ./map_localization.py  --mode=filter  --out_dir=results/radar/ ${cfg} || exit 1
    done
  done
fi
echo "Finished filter"
echo "Elapsed Time (using \$SECONDS): $SECONDS seconds"

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

if [ $DO_PLOTRESULTS -ne 0 ]; then
   # ./map_localization.py --mode=convert_gps_data  --out_dir=results/radar/ $CFGFILES || exit 1
   # ./map_localization.py --mode=plot_trajectories --out_dir=results/radar/ $CFGFILES || exit 1
   ./map_localization.py --mode=compute_errors_ziyang --out_dir=results/radar/ $CFGFILES || exit 1 

 
fi
# if [ $DO_EXTRACTCROP -ne 0 ]; then
#  for cfg in $CROP_CFGFILES; do
#    for scl in $SCALES; do
#      $MPICMD --allow-run-as-root ./map_localization.py --mode=extract_modes  --out_dir=results/ ${cfg} || exit 1
#    done
#  done
# fi
# echo "Finished extract_modes"

# if [ $DO_COMPCROPERR -ne 0 ]; then
#  for scl in $SCALES; do
#    $MPICMD --allow-run-as-root ./map_localization.py --mode=convert_gps_data  --out_dir=results/ $CROP_CFGFILES || exit 1
#  done
# echo "Finished convert_gps_data"
#  for scl in $SCALES; do
#    $MPICMD --allow-run-as-root ./map_localization.py --mode=compute_errors  --out_dir=results/ $CROP_CFGFILES || exit 1
#  done
# echo "Finished compute_errors"
# fi


# if [ $DO_PLOTCROPERRS -ne 0 ]; then
#  $MPICMD --allow-run-as-root ./map_localization.py --mode=plot_errors $PLOT_PARAMS --out_dir=results/ "$SCALES" $CROP_CFGFILES || exit 1
# fi

# # # MPICMD="mpirun -n 1"
# echo "Start display"
# if [ $DO_DISPLAY -ne 0 ]; then
#  $MPICMD --allow-run-as-root ./map_localization.py  --mode=display --out_dir=results/ $DISP_CFGFILES || exit 1
# fi

