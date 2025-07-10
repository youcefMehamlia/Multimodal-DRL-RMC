
function run () {

ALGO1="AlwaysGreenBaseline"
ALGO2="FixedCycleBaseline"
ALGO3="AlineaDsBaseline"
ALGO4="PiAlineaDsBaseline"

EPISODES = 1

python evaluate.py -s $ALGO1 -n $EPISODES 

}