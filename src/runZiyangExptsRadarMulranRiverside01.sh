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

# echo "display map"
# if [ $DO_DISPLAY -ne 0 ]; then
#   ./map_localization.py  --mode=display_map --out_dir=results/ $DISP_CFGFILES || exit 1
# fi

SECONDS=0
if [ $DO_FILTERCROP -ne 0 ]; then
  for cfg in $CROP_CFGFILES; do
    for scl in $SCALES; do
      $MPICMD --allow-run-as-root python ./map_localization.py  --mode=filter --out_dir=results/ ${cfg} || exit 1
    done
  done
fi
echo "Finished filter"
echo "Elapsed Time (using \$SECONDS): $SECONDS seconds"


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

# echo "Start display"
# if [ $DO_DISPLAY -ne 0 ]; then
#  $MPICMD --allow-run-as-root ./map_localization.py  --mode=display --out_dir=results/ $DISP_CFGFILES || exit 1
# fi

