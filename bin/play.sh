#!/usr/bin/bash

function run () {

ALGO1="AlwaysGreenBaseline"
ALGO2="FixedCycleBaseline"
ALGO3="AlineaDsBaseline"
ALGO4="PiAlineaDsBaseline"



python3 play.py -player $ALGO2 -max_e 1 -log True -log_s 40

}

# cd ..

# source venv/bin/activate

run

# deactivate

# exit
