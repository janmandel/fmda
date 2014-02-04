#!/usr/bin/env bash
CMD="./fmda_auto.sh"
TID="fmda"
PIDFILE="$TID.pid"
DONEFILE="$TID.exitcode"
touch /home/mvejmelka/Projects/wrfx2/wksp/25b55327-2c43-4d62-beb9-a314ccf91c9f/fmda/fmda.stdout
touch /home/mvejmelka/Projects/wrfx2/wksp/25b55327-2c43-4d62-beb9-a314ccf91c9f/fmda/fmda.stderr
./fmda_auto.sh 1>> /home/mvejmelka/Projects/wrfx2/wksp/25b55327-2c43-4d62-beb9-a314ccf91c9f/fmda/fmda.stdout 2>> /home/mvejmelka/Projects/wrfx2/wksp/25b55327-2c43-4d62-beb9-a314ccf91c9f/fmda/fmda.stderr &
PID=$!
echo $PID `date +%s` > $PIDFILE
wait $PID
echo $? `date +%s` > $DONEFILE
