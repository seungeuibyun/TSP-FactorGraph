import os
import sys

SUMO_HOME = os.environ["SUMO_HOME"]
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import sumolib

net = sumolib.net.readNet("seongbuk_buffer.net.xml")

print("edges:", len(net.getEdges()))
print("boundary:", net.getBoundary())

for e in net.getEdges()[:20]:
    print(
        e.getID(),
        e.getName(),
        e.getLength(),
        e.getSpeed(),
        e.getShape()
    )