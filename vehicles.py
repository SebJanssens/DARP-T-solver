import numpy as np
import LoadData as dat
from requests import req
from scipy.optimize import linear_sum_assignment
from copy import deepcopy as cp
import random
import funcs


#Class describing all the vehicles
class Vehicles:
    def __init__(self, capacity, startingPos):
        # Variables that never change
        self.n = len(capacity)
        self.capacity = capacity
        self.startingPos = startingPos[:]

        # Current state of the vehicle
        self.cNode = startingPos[:]
        self.cTime = np.zeros(self.n)
        # Protects the entire route of vehicles from modifications during the destroy and the repair phase
        self.isProtected = np.full(self.n, False)
        # From the pairing algorithm
        self.pairedWith = np.full(self.n, None)
        #Non empty vehicles have a certificate of feasibility
        self.isEmpty = np.full(self.n, True)
        # Certificate of feasibility
        self.certFeasibility = [[]] * self.n

        # Memoization of candRoutes
        self.W = [[] for i in range(self.n)]

    def copy(self):
        cpVeh = Vehicles(self.capacity, self.startingPos)
        cpVeh.cNode = self.cNode.copy()
        cpVeh.cTime = self.cTime.copy()
        cpVeh.isEmpty = self.isEmpty.copy()
        cpVeh.isProtected = self.isProtected.copy()
        cpVeh.pairedWith = self.pairedWith.copy()
        cpVeh.certFeasibility = [i.copy() if i != [] else [] for i in self.certFeasibility]
        cpVeh.W = cp(self.W)
        return cpVeh

    # Vehicle - request pairing by solving assignment problem
    def pair(self, rejectedRequests):
        # Find indices of vehicles and requests that are available for pairing
        vehAvailable, = np.where(self.isEmpty)
        reqAvailable, = np.where(~req.isHandled)
        # Every element ij of vrCost contains the time it will take for vehAvailable i to pick up a reqAvailable j
        vrCost = np.maximum(dat.tMat[np.ix_(self.cNode[vehAvailable], req.P[reqAvailable])],
                            req.E[reqAvailable] - self.cTime[vehAvailable, None])

        # Detect pairings that would be infeasible and give them a large cost (inf not supported by linear_sum_assignment)
        vrCost[self.cTime[vehAvailable, None] + vrCost > req.ld[reqAvailable]] = dat.largeN
        vrCost[req.groupSize[reqAvailable] - self.capacity[vehAvailable, None] > 0] = dat.largeN

        # Pickup time matrix
        pickupTimeMat = self.cTime[vehAvailable, None] + vrCost

        # Soonest arrival for the busy vehicles
        soonestArrivalBusy = np.amin(
            dat.tMat[np.ix_(self.cNode[~self.isEmpty], req.P[reqAvailable])] + self.cTime[~self.isEmpty, None],
            axis=0) if not all(self.isEmpty) else np.inf
        # Find requests that can not be served anymore
        toReject = (np.maximum(np.minimum(soonestArrivalBusy, np.amin(pickupTimeMat, axis=0)), req.E[reqAvailable]) >
                    req.ld[reqAvailable])
        # If all vehicles are available, requests that do not have a feasible pairing should also be rejected
        if all(self.isEmpty):
            toReject |= np.all(vrCost == np.inf, axis=0)
        # Drop said requests
        req.isHandled[reqAvailable[toReject]] = True
        # Log your failures
        rejectedRequests += list(reqAvailable[toReject])

        # Remove dropped requests from cost matrix
        reqAvailable = reqAvailable[~toReject]
        vrCost = vrCost[:, ~toReject]
        pickupTimeMat = pickupTimeMat[:, ~toReject]

        # Solve rectangular assignment problem
        sel_veh, sel_req = linear_sum_assignment(vrCost)
        # Remove infeasible assignments
        validAssignments = vrCost[sel_veh, sel_req] != dat.largeN
        sel_veh = sel_veh[validAssignments]
        sel_req = sel_req[validAssignments]

        # Erase previous assignments
        self.pairedWith = np.array([None] * self.n)
        # Write new assignment
        self.pairedWith[vehAvailable[sel_veh]] = reqAvailable[sel_req]

        # vInds, rInds = vehAvailable[sel_veh], reqAvailable[sel_req]
        # depTimes = np.maximum(self.cTime[vInds], req.E[rInds] - dat.tMat[self.cNode[vInds], req.P[rInds]])#Return the index of the vehicle that starts moving first as well as the departure time
        # If there is no valid paring
        if len(sel_veh) == 0:
            return -1, np.inf
        earliest = np.argmin(pickupTimeMat[sel_veh, sel_req] + np.random.randint(0, dat.vehSelShakeVal, len(sel_req)))
        return vehAvailable[sel_veh[earliest]]#, pickupTimeMat[sel_veh[earliest], sel_req[earliest]]
        # earliest = np.argmin(depTimes)
        # return vehAvailable[sel_veh[earliest]], depTimes[earliest]

    #Selects the vehicle whose entire route will be completed in the repair phase. Returns -1 if it there are none
    def select(self, iterNum, rejectedRequests):
        #Number of unhandled requests
        nNotHandled = req.n - np.sum(req.isHandled)

        # Pair vehicles and requests and select vehicle (not everytime when there are a lot unhandled requests, it's costly)
        if np.random.rand() < 20/max(1, nNotHandled) or iterNum == 1:
            v1 = self.pair(rejectedRequests)
        else:
            #We now either select a vehicle with certificate of feasibility or an empty vehicle:
            #Select vehicle with certificate of feasibility that first starts moving
            if np.random.rand() < 0.5 and not all(self.isEmpty):
                tPickUp = dat.largeN
                for i in np.where(~self.isEmpty)[0]:
                    if self.certFeasibility[i].tSeq[1] <= tPickUp:
                        tDep = self.certFeasibility[i].tSeq[1]
                        v1 = i
            else:#Select empty vehicle with minimum pickup time
                #We use the previously computed pairing
                mintPickup = dat.largeN
                bestv = -1
                for v in range(self.n):
                    if (not self.isEmpty[v]) or self.pairedWith[v] is None or req.isHandled[self.pairedWith[v]]:
                        continue
                    tPickup = max(req.E[self.pairedWith[v]], self.cTime[v] + dat.tMat[self.cNode[v], req.P[self.pairedWith[v]]])
                    if tPickup > req.ld[self.pairedWith[v]]:
                        continue
                    if tPickup < mintPickup:
                        mintPickup = tPickup
                        bestv = v
                if bestv > -1:
                    v1 = bestv
                else:
                    #Turns out we have to recompute a pairing
                    v1, = self.pair(rejectedRequests)
                    #Recomputing the pairing did not select a vehicle
                    if v1 == -1 and not all(~veh.isEmpty):
                        #Select a vehicle that is not empty
                        v1 = np.where(~veh.isEmpty)[0][0]
        return v1



    # Finds a set of feasible and infeasible candidate routes for vInd
    # Initial route can be provided with cRo, set of candidate request can be provided with candR
    # and set of request that can never be part of the candidate routes can be provided with reqForbidden
    def candRoutes(self, vInd, cRo=None, candR=None, reqForbidden=None):
        # To avoid mutability issues, default values of W, candR and reqForbidden are not mutable
        if candR is None:
            candR = []
        if reqForbidden is None:
            reqForbidden = set()

        # List of complete candidate routes
        WComp = [self.certFeasibility[vInd]] if cRo is None else [cRo]

        # Boolean indicating candR wether candR was inputted
        candRProvided = (candR != [])

        # List of incomplete candidate routes
        WInc = []
        WIncCost = []

        # All requests related to the request transported by the initial route
        if not candRProvided:
            candR = set().union(*req.related[WComp[0].reqIn])
        # Also add to candR the requests on veh.cNode -> first req.P part of the route
        # Pick-up delay should not make route infeasible
        isCand = dat.tMat[self.cNode[vInd], req.P] + dat.tMat[req.P, WComp[0].nodeSeq[0]] - dat.tMat[
            self.cNode[vInd], WComp[0].nodeSeq[0]] < WComp[0].tSlackSeq[0]
        # Can be picked up without large waiting
        isCand = isCand & (WComp[0].t0 + dat.tMat[self.cNode[vInd], req.P] + dat.maxWait / 2 > req.E)
        # Append new candidates with the constraint that at one least of the requests in candR must be related to the added request
        candR |= {i for i in np.where(isCand)[0] if len(candR & req.fRelated[i]) > 0} - reqForbidden

        # Set of requests removed (used) from candR
        removedr = set()

        while len(candR) > 0 and len(WComp) < 7:
            # Now we will try to insert a new, randomly selected request in WComp[-1]
            r = random.sample(candR, 1)[0]
            # Remove r from candR
            candR.remove(r)
            removedr.add(r)

            # Don't consider handled requests or requests already in WComp[-1]
            if req.isHandled[r] or (r in WComp[-1].reqIn):
                continue

            cRo = WComp[-1].copy()
            # Can r can be feasibly inserted?
            if cRo.insertRequest(r):
                WComp.append(cRo)
                # We update candR
                if not candRProvided:
                    # Add requests related to request just added
                    candR |= req.related[r]
                    candR -= removedr  # Make sure a checked request has not been added again
                continue

            # Maybe we can just pick r up (and figure out the dropoff later on with a transfer) and insert in some route in W?
            mCand = WComp[np.random.randint(len(WComp))].copy()
            if mCand.insertRequest(r, dropItOff=False):
                # Cost of resulting complete route
                cost = funcs.cost(mCand)
                # Only keep best three
                if len(WInc) < 3:
                    WInc.append(mCand)
                    WIncCost.append(cost)
                else:
                    worstInd = np.argmin(WIncCost)
                    WInc[worstInd] = mCand
                    WIncCost[worstInd] = cost

        self.W[vInd] = WComp + WInc
        # self.WInsMat[vInd] = [self.W[vInd][i].insertionMatrix(range(dat.nCity)) for i in range(len(self.W[vInd]))]


veh = Vehicles(dat.capacity, dat.startingPos)
