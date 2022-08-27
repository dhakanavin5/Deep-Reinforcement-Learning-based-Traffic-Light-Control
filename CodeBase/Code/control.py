import os
import sys
import optparse
import subprocess
import random
import numpy as np
import keras
import datetime
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model


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


class DQNAgent:
    def __init__(self):
        self.gamma = 0.95   # discount rate
        self.epsilon = 0.1  # exploration rate
        self.learning_rate = 0.0002
        self.memory = deque(maxlen=200)
        self.model = self._build_model()
        self.action_size = 2

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(12, 12, 1))
        x1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_1)
        x1 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x1)
        x1 = Flatten()(x1)

        input_2 = Input(shape=(12, 12, 1))
        x2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_2)
        x2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x2)
        x2 = Flatten()(x2)

        input_3 = Input(shape=(2, 1))
        x3 = Flatten()(input_3)

        x = keras.layers.concatenate([x1, x2, x3])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        model.compile(optimizer=keras.optimizers.RMSprop(
            lr=self.learning_rate), loss='mse')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class SumoIntersection:

    def chooseMode(self, junctions):
        # This method uses fuzzy logic to determine the mode of operation(Fair, Priority, Emergency)
        mode = ["fair", "priority", "emergency"]
        currentDT = datetime.datetime.now()
        time = currentDT.hour
        wgMatrix = self.mapVehicleToWg(weight_matrix)
        averageLoad = self.calculateReward(wgMatrix)[0]/4
        averageQueueLength = traci.edge.getLastStepHaltingNumber('-133305531#1') + traci.edge.getLastStepHaltingNumber(
                        '-133305558#3') + traci.edge.getLastStepHaltingNumber('-133305531#2') + traci.edge.getLastStepHaltingNumber('-133305558#2')
        averageQueueLength = averageQueueLength/4

        # Define vehicleId's Of Emergency Vehicles
        emergencyVIds = []
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('-133305531#1')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('-133305558#3')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('-133305531#2')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('-133305558#2')

        # if emergency vehicle is present in any of the road just return emergency mode
        for v in vehicles_road1:
            if len(emergencyVIds) != 0 and  v in emergencyVIds:
                return mode[2]
        for v in vehicles_road2:
            if len(emergencyVIds) != 0 and v in emergencyVIds:
                return mode[2]
        for v in vehicles_road3:
            if len(emergencyVIds) != 0 and v in emergencyVIds:
                return mode[2]
        for v in vehicles_road4:
            if len(emergencyVIds) != 0 and v in emergencyVIds:
                return mode[2]


        fair = 0
        priority = 0
        # Rules for fuzzification (More rules can be added later)
        if (time > 6 and time < 10) or (time > 15 and time < 18):
            fair = fair + 1
        else:
            priority = priority + 1

        if averageQueueLength < 10 :
            fair = fair + 1
        else:
            priority = priority + 1
            fair = fair + 1

        if averageLoad < 10 :
            fair = fair + 1
        else:
            priority = priority + 1
            fair = fair + 1

        fair = fair/3
        priority = priority/3

        #Defuzzification
        if fair > priority and fair <= 0.5:
            return mode[0]

        if priority > fair and priority <= 0.5:
            return mode[1]

        if random.uniform(0,1) < 0.5:
            return mode[0]
        else:
            return mode[1]



    def mapVehicleToWg(self, wgMatrix):
        # Define weights of each vehicle before the simuation
        return wgMatrix

    def findDuration(self, turn, wgMatrix):
        total = self.calculateReward(wgMatrix)

        if (total[0] == 0):
            return 0
        return min(5,max(30,(total[turn+1]/total[0])*60))


    def calculateReward(self, wgMatrix):
        reward = 0
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('-133305531#1')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('-133305558#3')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('-133305531#2')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('-133305558#2')

        for v in vehicles_road1:
            if (len(weight_matrix) != 0 and v in wgMatrix):
                reward += wgMatrix[v]
                r4 += wgMatrix[v]

            else:
                reward = reward + 1
                r4 = r4 + 1

        for v in vehicles_road2:
            if (len(weight_matrix) != 0 and v in wgMatrix):
                reward += wgMatrix[v]
                r3 += wgMatrix[v]
            else:
                reward = reward + 1
                r3 = r3 + 1

        for v in vehicles_road3:
            if (len(weight_matrix) != 0 and v in wgMatrix):
                reward += wgMatrix[v]
                r2 += wgMatrix[v]
            else:
                reward = reward + 1
                r2  = r2 + 1


        for v in vehicles_road4:
            if (len(weight_matrix) != 0 and v in wgMatrix):
                reward += wgMatrix[v]
                r1 += wgMatrix[v]
            else:
                reward = reward + 1
                r1 = r1 + 1


        return [reward, r1, r2, r3, r4]



       
    def getState(self):
        positionMatrix = []
        velocityMatrix = []

        cellLength = 7
        offset = 11
        speedLimit = 14

        junctionPosition = traci.junction.getPosition('1467490850')[0]
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('-133305531#1')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('-133305558#3')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('-133305531#2')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('-133305558#2')
        for i in range(12):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(12):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)

        for v in vehicles_road1:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
            if(ind < 12):
                positionMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[2 - traci.vehicle.getLaneIndex(
                    v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road2:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
            if(ind < 12):
                positionMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[3 + traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition('1467490850')[1]
        for v in vehicles_road3:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
            if(ind < 12):
                positionMatrix[6 + 2 -
                               traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[6 + 2 - traci.vehicle.getLaneIndex(
                    v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road4:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[1] + offset)) / cellLength)
            if(ind < 12):
                positionMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[9 + traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        light = [0, 1]

        position = np.array(positionMatrix)
        position = position.reshape(1, 12, 12, 1)

        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 12, 12, 1)

        lgts = np.array(light)
        lgts = lgts.reshape(1, 2, 1)

        return [position, velocity, lgts]


if __name__ == '__main__':
    sumoInt = SumoIntersection()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run


    #sumoInt.generate_routefile()

    # Main logic
    # parameters
    episodes = 2000
    batch_size = 32

    tg = 10
    ty = 6
    agent = DQNAgent()
    try:
        agent.load('Models/reinf_traf_control.h5')
    except:
        print('No models found')

    for e in range(episodes):
        # DNN Agent
        # Initialize DNN with random weights
        # Initialize target network with same weights as DNN Network
        #log = open('log.txt', 'a')
        step = 0
        waiting_time = 0
        reward1 = 0
        reward2 = 0
        total_reward = reward1 - reward2
        stepz = 0
        action = 0

        sumoBinary = checkBinary('sumo-gui')
        sumoProcess = subprocess.Popen([sumoBinary, "-c", "Map/cross.sumocfg", "--tripinfo-output",
                                    "tripinfo.xml", "--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)

        traci.init(PORT)
        junctions = traci.trafficlight.getIDList()
        traci.trafficlight.setPhase("1467490850", 0)
        traci.trafficlight.setPhaseDuration("1467490850", 200)
        turn = 0
        # contains vehicle to its wg matrix
        weight_matrix = []

        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 7000:
            traci.simulationStep()
            state = sumoInt.getState()
            action = agent.act(state)
            mode = sumoInt.chooseMode(junctions)
            # map vehicle to wg based on mode of operation
            if mode == "priority" or mode == "emergency":
                sumoInt.mapVehicleToWg(weight_matrix)
            # state change
            if(action == 1):
                turn = (turn + 1) % 4
                reward1 = sumoInt.calculateReward(weight_matrix)[0]
                for i in range(16):
                    stepz += 1
                    traci.trafficlight.setPhase('1467490850', turn)
                    traci.trafficlight.setPhaseDuration('1467490850', sumoInt.findDuration(turn, weight_matrix))
                traci.simulationStep()
                reward2 = sumoInt.calculateReward(weight_matrix)[0]
                

	        # no state change
            if(action == 0):
                
               	reward1 = sumoInt.calculateReward(weight_matrix)[0]
                for i in range(16):
                    stepz += 1
                    traci.trafficlight.setPhase('1467490850', turn)
                    traci.trafficlight.setPhaseDuration('1467490850', sumoInt.findDuration(turn, weight_matrix))
     		    traci.simulationStep()
                reward2 = sumoInt.calculateReward(weight_matrix)[0]


            new_state = sumoInt.getState()
            reward = reward1 - reward2
            agent.remember(state, action, reward, new_state, False)
            # Randomly Draw 32 samples and train the neural network by RMS Prop algorithm
            if(len(agent.memory) > batch_size):
                agent.replay(batch_size)

        mem = agent.memory[-1]
        del agent.memory[-1]
        agent.memory.append((mem[0], mem[1], reward, mem[3], True))
        #log.write('episode - ' + str(e) + ', total waiting time - ' +
        #          str(waiting_time) + ', static waiting time - 338798 \n')
        #log.close()
        print('episode - ' + str(e) + ' total waiting time - ' + str(waiting_time))
        #agent.save('reinf_traf_control_' + str(e) + '.h5')
        traci.close(wait=False)

sys.stdout.flush()
