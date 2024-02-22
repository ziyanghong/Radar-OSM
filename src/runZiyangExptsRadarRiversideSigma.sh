#!/bin/bash

DO_FILTERCROP=1
DO_EXTRACTCROP=1
DO_COMPCROPERR=1
DO_PLOTCROPERRS=1
DO_DISPLAY=1
# MPICMD="mpirun -n 12"
MPICMD="mpirun -n 1"

# MPICMD=""

#SCALES="0.125 0.25 0.5 1.0 2.0"
SCALES="1.0"

#CROP_CFGFILES="configs/0*o.dcfg configs/10*o.dcfg"
#CROP_CFGFILES="configs/0[0235789]*o.dcfg"
CROP_CFGFILES="configs/riverside01.dcfg"
DISP_CFGFILES="configs/riverside01.dcfg"
#PLOT_PARAMS="--img_format=png"
PLOT_PARAMS="--img_format=pdf"
DYNPARAMFILE=params/riverside_dynamics_params.p

# echo "display map"
# if [ $DO_DISPLAY -ne 0 ]; then
#   ./map_localization.py  --mode=display_map --out_dir=results/ $DISP_CFGFILES || exit 1
# fi

SECONDS=0
if [ $DO_FILTERCROP -ne 0 ]; then
  for cfg in $CROP_CFGFILES; do
    for scl in $SCALES; do
      $MPICMD --allow-run-as-root python ./map_localization.py  --mode=convert_gps_data --out_dir=results/ ${cfg} || exit 1
    done
  done
fi

SECONDS=0
if [ $DO_FILTERCROP -ne 0 ]; then
  for cfg in $CROP_CFGFILES; do
    for scl in $SCALES; do
      $MPICMD --allow-run-as-root python ./map_localization.py  --mode=fit_motionmodel --out_dir=results/ ${DYNPARAMFILE} || exit 1
    done
  done
fi
echo "Finished compute sigma"
echo "Elapsed Time (using \$SECONDS): $SECONDS seconds"

