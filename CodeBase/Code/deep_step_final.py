import os
import sys
import optparse
import subprocess
import random
import numpy as np
import time
from Dqn import Learner


try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
PORT = 8873
import traci

def makemap(TLIds):
    maptlactions = []
    n_phases = list_of_n_phases(TLIds)
    for n_phase in n_phases:
        mapTemp = []
        if len(maptlactions) == 0:
            for i in range(n_phase):
                if i%2 == 0:
                    maptlactions.append([i])
        else:
            for state in maptlactions:
                for i in range(n_phase):
                    if i%2 == 0:
                        mapTemp.append(state+[i])
            maptlactions = mapTemp
    return maptlactions

def list_of_n_phases(TLIds):
    n_phases = []
    for light in TLIds:
        print("1: " + traci.trafficlight.getRedYellowGreenState(light))
        n_phases.append(int((len(traci.trafficlight.getRedYellowGreenState(light)) ** 0.5) * 2))
    return n_phases


def get_state(detectorIDs):
    state = []
    for detector in detectorIDs:
        speed = traci.inductionloop.getLastStepMeanSpeed(detector)
        state.append(speed)
    for detector in detectorIDs:
        veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
        state.append(veh_num)
    state = np.array(state)
    state = state.reshape((1, state.shape[0]))
    return state


def calc_reward(state, next_state):
    rew = 0
    lstate = list(state)[0]
    lnext_state = list(next_state)[0]
    for ind, (det_old, det_new) in enumerate(zip(lstate, lnext_state)):
        if ind < len(lstate)/2:
            rew += 1000*(det_new - det_old)
        else:
            rew += 1000*(det_old - det_new)

    return rew


def main():
    # Control code here
    sumoBinary = checkBinary('sumo-gui')
    sumoProcess = subprocess.Popen([sumoBinary, "-c", "Map/cross.sumocfg", "--tripinfo-output",
                                    "tripinfo.xml", "--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)

    traci.init(PORT)
    TLIds = traci.trafficlight.getIDList()
    actionsMap = makemap(TLIds)
    detectorIDs = traci.inductionloop.getIDList()
    state_space_size = traci.inductionloop.getIDCount()*2
    action_space_size = len(actionsMap)
    agent = Learner(state_space_size, action_space_size, 0.0) 
    state = get_state(detectorIDs)                    
    total_reward = 0
    simulationSteps = 0
    while simulationSteps < 1000:
        for i in range(2):
            traci.simulationStep()
            time.sleep(0.4)         
        simulationSteps += 2
        next_state = get_state(detectorIDs)
        reward = calc_reward(state, next_state)
        total_reward += reward
        sum = 0
        print(total_reward + random.randint(0,9))
        state = next_state
    traci.close()
    
if __name__ == '__main__':
    main()
