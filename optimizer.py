import time
import numpy as np
from funcs import *
import random
from vehicles import veh
from requests import req
from solution import Solution

print(dat.name, dat.transEnabled)
print()

#Set seeds
np.random.seed(0)
random.seed(0)


#Default values
switchSA = False
rejectedRequests0 = []
#Loads existing solution
try:
    prevSol = np.load('bestSolFixed/bestSol_'+dat.name+'_nt.npz')
    prevSol.allow_pickle=True
    newSol = prevSol['bestSol'].item()
    veh = prevSol['vehicles'].item()
    rejectedRequests0 = list(prevSol['rejectedRequests'])
    req.isHandled |= True
    switchSA=True
except:
    pass

#Solution we are building (S in thesis report)
newSol = Solution(rejectedReqs=rejectedRequests0)

#Last accepted solution (S_a in the thesis report)
oldSol = Solution(None)

#We sometimes accept newSol even though it is worse than than oldSol to avoid being stuck in a local minima.
#Best solution found (ever)
bestSol = Solution(None)
bestVeh = None

#Repair and destroy counder
nRepairDestroy = 0

#Initialize temperature related variables
T = 0.00001
t0 = time.time()
tLastBestSol = 0
tLastAccepted = 0
tIter0 = 0

#Disables transfers during repair
noTransRepair = True



#Log values for debugging
scoreLog = []
nTransLog = []
timeLog = []
tempLog = []
acceptedLog = []
acceptedIndLog = []


while time.time() - t0 - tLastBestSol < 30000000 and time.time() - t0 - tLastBestSol < 40*60:
    #Repair the solution
    newSol.repair(nRepairDestroy, noTransRepair=noTransRepair)
    #Run solution validity checker (optional)
    newSol.check()
    # Time at end of first repair
    if nRepairDestroy == 0:
        tIter0 = time.time() - t0

    # If new solution is better than best known solution
    if newSol.score < bestSol.score:
        #Reset temperature
        T = 0.000001
        #Store solution and context
        bestSol = newSol.copy()
        bestVeh = veh.copy()
        print('NEW BEST SOLUTION!', bestSol.score, 'Fails', newSol.rejectedReqs, 'nTrans', np.sum([sum(np.array(bestSol.transferTo[i]) != None) for i in range(veh.n)]))
        print()
        tLastBestSol = time.time() - t0

    #Update temperature
    if switchSA:
        #Quick Reheat
        T = max(0.00001, np.e ** (-.04 * ((time.time() - t0 - tIter0) * 3 + 500)))
    else:
        T = max(0.00001, np.e ** (-.04 * ((time.time() - t0 - tIter0) / 15 + 15)))

    #Update logs
    scoreLog.append(newSol.score)
    nTransLog.append(np.sum([sum(np.array(newSol.transferTo[i]) != None) for i in range(veh.n)]))
    timeLog.append(time.time() - t0)
    tempLog.append(T)
    acceptedLog.append(False)

    #If newSol is accepted
    if (newSol.score not in [scoreLog[i] for i in acceptedIndLog[-20:]]) and (newSol.score < oldSol.score or (newSol.score != oldSol.score and np.e**(-150*(newSol.score-oldSol.score - 0.15/req.n*(1+4*T)*max(0, newSol.nTransfers - oldSol.nTransfers))/T) > np.random.rand())):
        #Copy solution and context
        oldSol = newSol.copy()
        oldcNode = veh.cNode.copy()
        oldcTime = veh.cTime.copy()

        if newSol.score < oldSol.score:
            tLastAccepted = time.time() - t0
        acceptedIndLog.append(nRepairDestroy)
        acceptedLog[-1] = True

    #Restore oldSol and its context
    newSol = oldSol.copy()
    veh.cNode = oldcNode.copy()
    veh.cTime = oldcTime.copy()

    if (timeLog[-1] - tLastAccepted > 15*60 and timeLog[-1] - tLastBestSol < 20*60):
        #Switch/reset cooling schedule
        switchSA = True
        tIter0 = timeLog[-1]
        tLastAccepted = timeLog[-1]

    #No transfer optimization: we are not considering transfers in this iteration
    noTransRepair = np.random.rand() < .6

    #Determine Q
    Q = T
    if np.random.rand()<.8:
        Q = np.random.uniform(0.0, .2) if dat.transEnabled and not noTransRepair else  np.random.uniform(0.0, .4)
        if np.random.rand() < .5:
            Q = np.random.uniform(0.0, 0.05) if dat.transEnabled and not noTransRepair else  np.random.uniform(0.0, 0.1)

    #Destroy solution
    newSol.destroy(noTransRepair, Q)

    nRepairDestroy += 1


#Store solution
tmp = '_t' if dat.transEnabled else '_nt'
np.savez('solutions/'+dat.name+tmp, bestSol=bestSol, vehicles=bestVeh, timeLog=timeLog, tempLog=tempLog, scoreLog=scoreLog, acceptedLog=acceptedLog, nTransLog=nTransLog)

