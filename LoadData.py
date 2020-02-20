#Instance data and settings is gathered by this script

import numpy as np


#SETTINGS

#Load instance name
#np.savez('Instances/instToSolve', name='inst_150_70_4', nt=True)
name = np.load('Instances/instToSolve.npz')['name'].item()
transEnabled = np.load('Instances/instToSolve.npz')['nt'].item()

#Large constant
largeN = 1e10

#Randomization parameters
maxDetour = 40#minutes
maxWait = 25#minutes
timeClose = 20#minutes
transferInsDelay = 20#minutes
vehSelShakeVal = 15#In minutes


#DATA

#Data shared by all instances
timeScale = 240
#City names
cities = np.array(np.load('dat/arcs.npz')['cities'])
cityDict = {name.split(',')[0]: i for i, name in enumerate(cities)}
nCity = len(cities)

#Distance matrix
dat = np.load('dat/osmr.npz')
dat.allow_pickle = True
tMat = np.round(dat['duration']/60).astype('int')
dat.close()

#Load MDS coordinates
mdsLoc = np.load('dat/mdsLoc.npz')['loc']

#Data specific to the instance specified by instToSolve
dat = np.load('Instances/' + name + '.npz')
#Requests
earliest = dat['earliest']
latest = dat['latest']
groupSize = dat['groupSize']
pickup = dat['ori']
dropoff = dat['dest']
#Vehicles
capacity = dat['vCap']
startingPos = dat['vPos']
dat.close()

