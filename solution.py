import numpy as np
import LoadData as dat
from LoadData import tMat
from vehicles import veh
from requests import req
from functools import reduce
from copy import deepcopy as cp
from route import Route
from funcs import *
from numpy.random import uniform
from random import sample

class Solution:
    def __init__(self, emptySol=False, rejectedReqs=[]):
        #Score and number of transfers of empty solution
        self.score = dat.largeN
        self.nTransfers = 0
        if emptySol:
            return

        #Number of vehicles
        self.nVeh = veh.n
        #List of rejected requests
        self.rejectedReqs = rejectedReqs

        #For every vehicle, the sequence of nodes visited by the vehicle
        self.nodeSeq = [[i] for i in veh.startingPos]
        # For every vehicle and every node in nodeSeq:
        # -the request that got in the vehicle
        self.gotIn = [[None] for i in range(self.nVeh)]
        # -the request that got out of the vehicle
        self.gotOut = [[None] for i in range(self.nVeh)]
        # -the time at which the node was left
        self.tSeq = [[0] for i in range(self.nVeh)]
        # -boolean indicating if the vehicle was empty when leaving the node
        self.isEmptySeq = [[True] for i in range(self.nVeh)]
        # -the vehicle to which the transferred request was transferred
        self.transferTo = [[None] for i in range(self.nVeh)]
        # -the vehicle the the transferred request comes from
        self.transferFrom = [[None] for i in range(self.nVeh)]
        # -a list of the requests inside the vehicle when leaving the node
        self.reqInSeq = [[[]] for i in range(self.nVeh)]


        # Performance of the entire routes
        self.vPerf = np.array([dat.largeN] * self.nVeh)
        # Booleans indicating whether the entire route of a vehicle can be modified by the repair/destroy operator
        self.canBeModified = np.full(self.nVeh, False)

    # A lot faster than deepcopy
    def copy(self):
        cpSol = Solution(rejectedReqs=self.rejectedReqs[:])
        cpSol.score = self.score
        cpSol.nVeh = self.nVeh
        cpSol.nodeSeq = [i[:] for i in self.nodeSeq]
        cpSol.gotIn = [i[:] for i in self.gotIn]
        cpSol.gotOut = [i[:] for i in self.gotOut]
        cpSol.tSeq = [i[:] for i in self.tSeq]
        cpSol.isEmptySeq = [i[:] for i in self.isEmptySeq]
        cpSol.transferTo = [i[:] for i in self.transferTo]
        cpSol.transferFrom = [i[:] for i in self.transferFrom]
        cpSol.reqInSeq = cp(self.reqInSeq)
        cpSol.vPerf = self.vPerf.copy()
        cpSol.canBeModified = self.canBeModified.copy()
        cpSol.nTransfers = self.nTransfers
        return cpSol

    #Repairs a partial solution to a solution
    def repair(self, nRepairDestroy, noTransRepair=False):
        repairIterNum = 0
        #While the solution is not entirely repaired
        while not (all(req.isHandled) and all(veh.isEmpty)):
            repairIterNum += 1
            print(req.n - np.sum(req.isHandled)) if nRepairDestroy == 0 else None

            # Selects a vehicle and rejects requests that can not be served anymore
            v1 = veh.select(repairIterNum, self.rejectedReqs)

            # If veh.select did not select a vehicle
            if v1 == -1:
                # all requests should now be handled (veh.select can reject requests)
                if not all(req.isHandled):
                    raise RuntimeError("No vehicle selected and requests still unhandled!")
                # We are done repairing
                break

            # Generation of candidate routes for v1
            if veh.isEmpty[v1]:
                # Generation of initial route for candRoutes
                # Make vehicle wait before going to request if needed
                veh.cTime[v1] = max(veh.cTime[v1],
                                    req.E[veh.pairedWith[v1]] - tMat[veh.cNode[v1], req.P[veh.pairedWith[v1]]])
                roIni = Route(veh.pairedWith[v1], veh.cTime[v1], veh.cNode[v1], v1)
                if not roIni.compute():
                    raise RuntimeError('Basic route does not seem feasible!')
                veh.candRoutes(v1, roIni)
            else:
                veh.candRoutes(v1)
            # Cost of the candidate routes for v1
            WCosts = [uniform(.7, 1.3) * cost(w1) if not w1.isIncomplete else np.inf for w1 in veh.W[v1]]
            bestw1 = np.argmin(WCosts)
            self._bestCost = WCosts[bestw1]

            # Search for transfer
            # Best option (between a route in W and a transfer)
            self._bestIsTransfer = False
            self._stopTransferSearch = False
            for w1Ind in sample(range(len(veh.W[v1])), k=len(veh.W[v1])):  # Every candidate route
                # If search for transfer is disabled
                if (not dat.transEnabled) or noTransRepair:
                    break

                # Shortcut notation
                w1 = veh.W[v1][w1Ind]

                # List of candidate nodes for a transfer (based on route w1) and corresponding insertion position(s) into w1
                w1TransCand, w1InsPos = reformatDuplicates(*np.where(w1.insertionMatrix(range(dat.nCity)) <
                                                                     uniform(.8, 1.2) * np.minimum(dat.transferInsDelay, w1.tSlackSeq)))
                if len(w1TransCand) == 0:
                    continue

                # Can another vehicle assist w1?
                for v2 in sample(range(veh.n), k=min(veh.n, 30)):
                    # If v2 is v1 or v2 is too far in time to help w1
                    if v2 == v1 or (veh.cTime[v2] > w1.tSeq[-1] and not self.canBeModified[v2]):
                        continue

                    # If we consider organising a transfer between w1 and the entire route of v2
                    if self.canBeModified[v2] and veh.cTime[v2] > (w1.tSeq[-1] + uniform(-15, 15)):
                        if v1 in self.transferTo[v2] or v1 in self.transferFrom[v2]:
                            continue

                        # We skip with 50% chance if v2 is already involved in some transfer
                        if np.random.rand() < 0.5:
                            isInTransfer = False
                            for tt, tf in zip(self.transferTo[v2], self.transferFrom[v2]):
                                if tt is not None or tf is not None:
                                    isInTransfer = True
                                    break
                            if isInTransfer:
                                continue

                        # Indices of the relevant portion of the solution
                        indBegin = np.searchsorted(self.time[v2], w1.t0, 'right') - 1
                        indEnd = np.searchsorted(self.time[v2], w1.tSeq[-1])
                        if indEnd - indBegin < 1:
                            continue

                        # We create a fake route to input to getTransfer
                        w2 = Route([], dat.largeN, dat.largeN, v2)
                        w2.nodeSeq = self.nodeSeq[v2][indBegin:indEnd + 1]
                        w2.tSeq = self.time[v2][indBegin:indEnd + 1]
                        w2InsMat = tMat[np.ix_(w2.nodeSeq[:-1], w1TransCand)].T + tMat[
                            np.ix_(w1TransCand, w2.nodeSeq[1:])] - tMat[w2.nodeSeq[:-1], w2.nodeSeq[1:]]
                        # Other input for getTransfer
                        transCandInd, w2InsPos = reformatDuplicates(
                            *np.where(w2InsMat < uniform(.8, 1.2) * dat.timeClose))

                        #Compute transfer between w1 and v2
                        self.__getTransfer(w1, w2, cost(veh.W[v1][bestw1]), transCandInd, w1TransCand, w1InsPos, w2InsPos, indBegin=indBegin, modifySol=True)
                        continue

                    # If we're not modifying the solution, v2 does not have a route yet and we generate candidate routes for v2
                    # Distance between v2 and w1 (Euclidian distance with mds coordinates)
                    distv2w1 = w1.distPoint2Route([veh.cNode[v2]])[0][0]
                    # If v2 is too far and too much in the future
                    if veh.cTime[v2] + distv2w1 >= w1.tSeq[-1]:
                        continue

                    # Request that might be interesting to bring to transfer point: transCand with req in w1, not handled or in w1, can be picked up in time
                    candR = set([i for i in set.union(*[req.transCand[j] for j in w1.reqIn]) if
                                 (not req.isHandled[i]) and (not i in w1.reqIn) and veh.capacity[v2] >= req.groupSize[
                                     i] and
                                 (req.E[i] - dat.maxWait <= veh.cTime[v2] + tMat[veh.cNode[v2], req.P[i]] <= req.ld[
                                     i])])
                    # Keep at most 25 elements in candR
                    candR = sample(candR, k=min(len(candR), 25))

                    # If there are not enough requests for a transfer
                    if len(w1.reqIn) + len(candR) < 2:
                        continue

                    tryEmptyVi = False
                    # If v2 is not already assigned a route
                    if veh.certFeasibility[v2] == []:
                        # We also want to try no route for v2
                        tryEmptyVi = True

                        # Select a request in candR to create an initial route for candRoutes
                        selecR = -1
                        for r in candR:
                            # Distance pickup of r to w1 (Euclidian distance with mds)
                            distP2w1 = w1.distPoint2Route([req.P[r]])
                            # "on the way" check
                            if tMat[veh.cNode[v2], req.P[r]] + distP2w1[0][0] - distv2w1 > uniform(.8,
                                                                                                   1.2) * dat.timeClose:
                                continue
                            # Direction check
                            if (distP2w1[2][0] - dat.mdsLoc[req.P[r]]) @ (
                                    dat.mdsLoc[req.D[r]] - dat.mdsLoc[req.P[r]]) < 0:
                                continue
                            selecR = r
                            break
                        if selecR == -1:
                            continue
                        # Create the initial route for candRoutes
                        w2Init = Route(selecR, veh.cTime[v2], veh.cNode[v2], v2)
                        if not w2Init.compute():
                            raise RuntimeError('Initial route generation failed!')

                    else:
                        w2Init = veh.certFeasibility[v2]

                    # Generate some more candidate routes and compute their costs
                    veh.candRoutes(v2, W=w2Init, candR=candR, reqForbidden=set(w1.reqIn))
                    v2Costs = [uniform(.9, 1.1) * cost(w2) if not w2.isIncomplete else np.inf for w2 in veh.W[v2]]
                    bestv2 = np.argmin(v2Costs)
                    # Cost of v1 and v2 each performing their best routes (individually)
                    costNoTransfer = cost(veh.W[v1][bestw1], veh.W[v2][bestv2])

                    # For the candidate routes of v2
                    for w2Ind in sample(range(len(veh.W[v2]) + tryEmptyVi), k=len(veh.W[v2]) + tryEmptyVi):
                        # w2Ind equal to len(veh.W[v2]) means we try empty route
                        emptyRouteV2 = w2Ind == len(veh.W[v2])

                        # If we consider empty route for v2
                        if emptyRouteV2:
                            w2 = Route([], veh.cTime[v2], veh.cNode[v2], v2)
                            w2InsMat = tMat[w2.n0, w1TransCand]
                        else:
                            w2 = veh.W[v2][w2Ind]
                            w2InsMat = veh.W[v2][w2Ind].insertionMatrix(range(dat.nCity))[w1TransCand]

                        # Continue if not enough requests for transfers
                        if len(w1.reqIn) + len(w2.reqIn) < 2:
                            continue

                        # Only keep nodes shared with v1
                        if emptyRouteV2:
                            transCandInd, = np.where(w2InsMat < uniform(.8, 1.2) * dat.transferInsDelay)
                        else:
                            transCandInd, w2InsPos = reformatDuplicates(
                                *np.where(w2InsMat < uniform(.8, 1.2) * np.minimum(dat.transferInsDelay, w2.tSlackSeq)))

                        if len(transCandInd) == 0:
                            continue

                        #Compute the transfer between w1 and w2
                        self.__getTransfer(w1, w2, costNoTransfer, transCandInd, w1TransCand, w1InsPos, w2InsPos, emptyRouteV2=emptyRouteV2)

                        if self._stopTransferSearch:
                            #Breaks w2 loop
                            break

                    if self._stopTransferSearch:
                        # breaks v2 loop
                        break

                if self._stopTransferSearch:
                    # breaks w1 loop
                    break

            # Solution update

            # Adding the transfer to the solution if the entire route of v1 or v2 is destroyed by propagation when destroying v2's entire route
            transferUpdateFailed = False
            #If we're adding a transfer to the solution
            if self._bestIsTransfer:
                #If v2's entire route was modify
                if self._tBest['modifySol']:
                    v2 = self._tBest['v2']
                    # Destroy v2's entire route
                    if self.destroyRoute(v2, self._tBest['v2Ins'] + 1, noDropoff=True, nUnitsSpec=[[v1, len(self.nodeSeq[v1])], [v2, self._tBest['v2Ins']]]):
                        #The destruction is succesful
                        # Update v1 and v2
                        self.addTransferRoute(v1, v2, self._tBest['w1'], self._tBest['tNode'], self._tBest['v1Ins'], self._tBest['dRoV1'])
                        self.addTransferRoute(v2, v1, [], self._tBest['tNode'], self._tBest['v2Ins'], self._tBest['dRoV2'], veh)
                        # Rejected request should be available again (because of the destruction of v2's entire route)
                        req.isHandled[self.rejectedReqs] = False
                        self.rejectedReqs = []

                    else:
                        #The destroy operator recursed on v2 or on v1 and the transfer can not be executed anymore :(
                        print('Destroy operator made transfer impossible', repairIterNum, nRepairDestroy, v1, v2)
                        transferUpdateFailed = True
                else:
                    # Update v1 and v2
                    self.addTransferRoute(v1, self._tBest['v2'], self._tBest['w1'], self._tBest['tNode'], self._tBest['v1Ins'], self._tBest['dRoV1'])
                    self.addTransferRoute(self._tBest['v2'], v1, self._tBest['w2'], self._tBest['tNode'], self._tBest['v2Ins'], self._tBest['dRoV2'])

            #If the best option is without transfer or if the transfer failed
            if not self._bestIsTransfer or transferUpdateFailed:
                self.addRoute(veh.W[v1][bestw1], v1)

        #Compute objective function
        self.score = self.objFun(len(self.rejectedReqs))

    # Given 2 routes, finds a transfer
    def __getTransfer(self, w1, w2, costNoTransfer, transCandInd, w1TransCand, w1InsPos, w2InsPos, indBegin=None, emptyRouteV2=False, modifySol=False):
        # For every transfer node candidate
        for ti in sample(range(len(transCandInd)), k=min(5, len(transCandInd))):
            tNode = w1TransCand[transCandInd[ti]]

            # Time at which tNode is reached by v1 and v2
            v1Arrival = np.array([w1.tSeq[ip] + tMat[w1.nodeSeq[ip], tNode] for ip in w1InsPos[transCandInd[ti]]])
            v2Arrival = np.array([w2.nodeSeq[ip] + tMat[w2.nodeSeq[ip], tNode] for ip in w2InsPos[ti]]) if \
                not emptyRouteV2 else np.array([w2.t0 + tMat[w2.n0, tNode]])

            dtArrival = abs(v1Arrival - v2Arrival[:, None])  # dtArrival[i, j] = v1Arrival[j] - v2Arrival[i]
            # We take the insertion position that minimizes the time vehicle have to wait for each other
            # Array is completely reversed (should be efficient, only stride is modified) so that argmin returns the indx of the last occurence of the min
            v2Ins, v1Ins = np.unravel_index(np.argmin(dtArrival[::-1, ::-1]), dtArrival.shape)
            tTransV2 = v2Arrival[-v2Ins - 1]
            tTransV1 = v1Arrival[-v1Ins - 1]
            # Bring back index to route index
            v2Ins = w2InsPos[ti][len(v2Arrival) - 1 - v2Ins] if not emptyRouteV2 else 0
            v1Ins = w1InsPos[transCandInd[ti]][len(v1Arrival) - 1 - v1Ins]

            # Make sure they meet within maxWait
            if abs(tTransV2 - tTransV1) > dat.maxWait:
                continue

            # Find set of requests in vehicle at tNode and make sure there are at least two
            reqV1 = w1.rInSeq(v1Ins)
            if modifySol:
                reqV2 = self.reqInSeq[w2.v][indBegin + v2Ins]
            else:
                reqV2 = w2.rInSeq(v2Ins) if not emptyRouteV2 else ([], [])

            #All the requests involved in the transfer
            allReq = np.concatenate((reqV1, reqV2))
            #If there not enough requests for a transfer
            if len(allReq) < 2:
                continue

            # Time at which the transfer will take place
            tTransfer = max(tTransV1, tTransV2)
            # If one request's latest departure from tNode is before the time at which the transfer happens
            if np.min(req.L[allReq] - tMat[tNode, req.D[allReq]]) <= tTransfer:
                continue

            #For every candidate dropoff routes
            for dRoV1, dRoV2 in getDropoffRoutes(w1.v, w2.v, tNode, tTransV1, tTransV2, tTransfer, reqV1, reqV2, allReq):
                if np.sum(req.groupSize[dRoV1.reqIn]) > veh.capacity[w1.v] or np.sum(req.groupSize[dRoV2.reqIn]) > \
                        veh.capacity[w2.v]:
                    raise RuntimeError('Capacity violation')

                #Cost of the transfer
                costTransfer = cost(w1, w2, dRoV1, dRoV2)
                if (not modifySol and costTransfer <= self._bestCost and costTransfer <= uniform(1, 1.1) * costNoTransfer) or \
                    (modifySol and costTransfer <= self._bestCost and np.random.rand() <= [1, .7, .3, .1, 0, 0][len(reqV2)]):
                    self._bestCost = costTransfer
                    self._bestIsTransfer = True
                    #Save the transfer in memory
                    self._tBest = {'w1': w1, 'v2': w2.v, 'w2': w2, 'v1Ins': v1Ins, 'v2Ins': v2Ins + (indBegin if modifySol else 0), 'dRoV1': dRoV1,
                             'dRoV2': dRoV2, 'tNode': tNode, 'modifySol': modifySol}
                    if np.random.rand() < .2:
                        # Stop the search
                        self._stopTransferSearch = True
                        break

            if self._stopTransferSearch:
                # break ti loop
                break

    #Append a route to an entire route
    def append(self, vInd, nodeSeq, tSeq, gotIn, gotOut, isEmptySeq, transferTo, transferFrom):
        #New values are appended
        self.nodeSeq[vInd] += list(nodeSeq)
        self.gotIn[vInd] += list(gotIn)
        self.gotOut[vInd] += list(gotOut)
        self.tSeq[vInd] += list(tSeq)
        self.isEmptySeq[vInd] += list(isEmptySeq)
        self.transferTo[vInd] += transferTo
        self.transferFrom[vInd] += transferFrom
        #New reqInSeq is derived
        for rIn, rOut in zip(gotIn, gotOut):
            newReqIn = self.reqInSeq[vInd][-1][:]
            if rOut is not None:
                newReqIn.remove(rOut)
            if rIn is not None:
                newReqIn.append(rIn)
            self.reqInSeq[vInd].append(newReqIn)

    # Add a route (containing no transfer) to the solution
    def addRoute(self, ro, vInd):
        gotIn = ro.reqFlow(True)
        gotOut = ro.reqFlow(False)
        nbElem = len(gotOut)
        # Update newSol
        self.append(vInd, ro.nodeSeq, ro.tSeq, gotIn, gotOut, [len(ro.reqIn0) > 0] * nbElem, [None] * nbElem, [None] * nbElem)


        veh.cTime[vInd] = ro.tSeq[-1]
        veh.cNode[vInd] = ro.nodeSeq[-1]
        veh.certFeasibility[vInd] = []
        veh.isEmpty[vInd] = True
        req.isHandled[ro.reqIn] = True

    #Add a single route containing a transfer to the solution
    def addTransferRoute(self, vInd, otherVInd, pRoute, tNode, vIns, dRoute):
        #If there is no dropoff route
        if len(dRoute.reqInSeq) == 0:
            veh.certFeasibility[vInd] = []
            veh.isEmptySeq[vInd] = True
        else:
            #Assign dropoff route as certificate of feasibility
            veh.certFeasibility[vInd] = dRoute
            veh.certFeasibility[vInd].reqIn0 = set(dRoute.reqInSeq)
            dRoute.toPickup &= False
            veh.isEmptySeq[vInd] = False

        #Update vehicle's position and time
        veh.cNode[vInd] = dRoute.startNode
        veh.cTime[vInd] = dRoute.t0
        #Update request status
        req.isHandled[dRoute.reqInSeq] = True

        #PICKUP ROUTE
        #If there is a pickup route (there is none if transfer is inserted in an entire route)
        if pRoute != []:
            #Prepare the variables that need to be sent to append
            gotInPickup = pRoute.reqFlow(True)[:vIns + 1] if pRoute.isComputed else [None]
            reqPickup = [i for i in gotInPickup if i != None]

            # Update status of requests that have picked up pRoute (if they were dropped off durind the pickup route, they are not in dRoute!)
            req.isHandled[reqPickup] = True

            # Route until transfer point insertion (not included)
            self.append(vInd, pRoute.nodeSeq[:vIns + 1], pRoute.tSeq[:vIns + 1], gotInPickup,
                            pRoute.reqFlow(False)[:vIns + 1] if pRoute.isComputed else [None],
                            [True] * (vIns + 1), transferTo=[None] * (vIns + 1), transferFrom=[None] * (vIns + 1))

            reqAtTrans = pRoute.rInSeq[vIns] if len(pRoute.reqInSeq) > 0 else []
        else:
            reqAtTrans = self.reqInSeq[vInd][vIns]

        #TRANSFER
        #Requests that got in and out at the transfer
        gotIn = list(set(dRoute.reqInSeq) - set(reqAtTrans))
        gotOut = list(set(reqAtTrans) - set(dRoute.reqInSeq))
        # At most one request in and one out per unit of solution => number of necessary units
        nUnits = max(len(gotIn), len(gotOut))
        nFillIn = (nUnits - len(gotIn))
        nFillOut = (nUnits - len(gotOut))
        # Transfer
        self.append(vInd, [tNode] * nUnits, [veh.cTime[vInd]] * nUnits, gotIn + [None] * nFillIn, gotOut + [None] * nFillOut, [True] * nUnits,
                         transferTo=[otherVInd] * len(gotOut) + [None] * nFillOut, transferFrom=[otherVInd] * len(gotIn) + [None] * nFillIn)

    #Objective function
    def objFun(self, nFails):
        #Total distance traveled by the vehicles
        totDist = 0
        for i in range(len(self.nodeSeq)):
            totDist += reduce(lambda s, j: s + tMat[self.nodeSeq[i][j - 1], self.nodeSeq[i][j]], range(len(self.nodeSeq[i])))
        return totDist/req.totTravelTime + .1*nFails

    #Evaluate the performance of every vehicle by assigning a value (time spent driving/request time carried) similar to cost to every entire route.
    def vehPerf(self):
        #Numerator
        timeDriving = np.zeros(self.nVeh)
        #Denominator
        tReqCarried = np.zeros(self.nVeh)
        #Number of transfers the vehicle is involved in
        nTransfers = np.full(self.nVeh, 0.)

        #For every entire route
        for v in range(self.nVeh):
            #The time the vehicle spent driving
            timeDriving[v] = reduce(lambda s, j: s + tMat[self.nodeSeq[v][j - 1], self.nodeSeq[v][j]], range(len(self.nodeSeq[v])))

            #Filling in tReqCarried[v]
            #Request currently inside the vehicle
            reqInSeq = []
            #nodeSeq index at which the jth request in reqInSeq got in the vehicle
            jGotIn = []
            for j in range(len(self.nodeSeq[v])):
                #If there is a transfer
                if self.transferFrom[v][j] is not None or self.transferTo[v][j] is not None:
                    nTransfers[v] += 1

                #if a request got in
                if self.gotIn[v][j] != None:
                    reqInSeq.append(self.gotIn[v][j])
                    jGotIn.append(j)

                #if a request got out
                rOut = self.gotOut[v][j]
                if rOut != None:
                    indx = reqInSeq.index(rOut)
                    # Formula from report
                    nIn, nOut = self.nodeSeq[v][jGotIn[indx]], self.nodeSeq[v][j]
                    if nIn != req.P[rOut] and nOut != req.D[rOut]:
                        tReqCarried[v] += tMat[nIn, req.D[rOut]] - tMat[nOut, req.D[rOut]]
                    else:
                        tReqCarried[v] += tMat[req.P[rOut], req.D[rOut]] - tMat[req.P[rOut], nIn] - tMat[
                            nOut, req.D[rOut]]
                    #Request is not inside anymore
                    reqInSeq.pop(indx)
                    jGotIn.pop(indx)

        self.vPerf = timeDriving / (tReqCarried + 0.1) + .1*nTransfers
        return self.vPerf, nTransfers != 0

    # Performs some feasibility checks and returns objective value
    def check(self):
        # #Computes request routes
        # self.postProcess()
        # for i in range(req.n):
        #     for j in range(1,len(self.rNodeSeq[i])):
        #         if self.rNodeSeq[i][j - 1] != self.rNodeSeq[i][j] and tMat[self.rNodeSeq[i][j - 1], self.rNodeSeq[i][j]] > self.rTSeq[i][j] - self.rTSeq[i][j - 1]:
        #             raise RuntimeError('Invalid move: teleportation', i, j)

        if not all(veh.isEmpty) or len([i for i in veh.certFeasibility if i != []]) > 0:
            raise RuntimeError('not all vehicles are empty')

        #Pickup and dropoff time of all requests
        pickUpTime = np.array([-1] * req.n)
        dropOffTime = np.array([-1] * req.n)
        #Logs which vehicle is serving a request
        servedBy = np.full(req.n, None)
        isPickedUp = np.full(req.n, False)
        for v in range(len(self.nodeSeq)):
            #Sequence len check
            if not (len(self.nodeSeq[v]) == len(self.tSeq[v]) == len(self.gotIn[v]) == len(self.gotOut[v]) == len(
                    self.isEmptySeq[v])) == len(self.transferTo) == len(self.transferFrom) == len(self.reqInSeq):
                raise RuntimeError('Sequences do not all have same size', v)

            #Content of the vehicle as we parse the solution
            content = []
            for j in range(len(self.gotIn[v])):
                # Vehicle teleportation
                if j > 0:
                    if self.nodeSeq[v][j - 1] != self.nodeSeq[v][j] and tMat[self.nodeSeq[v][j - 1], self.nodeSeq[v][j]] > \
                            self.tSeq[v][j] - self.tSeq[v][j - 1]:
                        raise RuntimeError('Invalid move: teleportation', v, j)

                #If a request is entering vehicle v
                if self.gotIn[v][j] is not None:\
                    #Request that entered the vehicle
                    r = self.gotIn[v][j]
                    #Update variables
                    isPickedUp[r] = True
                    content.append(r)
                    if req.P[r] == self.nodeSeq[v][j]:
                        if pickUpTime[r] != -1:
                            raise RuntimeError('Request already picked up', v, j)
                        pickUpTime[r] = self.tSeq[v][j]

                    #Pickup time check
                    if self.tSeq[v][j] < req.E[r]:
                        raise RuntimeError('Request picked up before earliest pickup time')

                    #Request did not get in at pickup and doesnt come from another vehicle
                    if req.P[r] != self.nodeSeq[v][j] and self.transferFrom[v][j] is None:
                        raise RuntimeError('Request not picked up at pickup')


                #If a request is leaving vehicle
                if self.gotOut[v][j] is not None:
                    #Request leaving the vehicle
                    r = self.gotOut[v][j]
                    try:
                        content.remove(r)
                    except:#Request was not in the vehicle
                        raise RuntimeError('Request not in vehicle dropped off', v)

                    #Request leaves the vehicle not at dropoff point and is not transfered to another vehicle
                    if req.D[r] != self.nodeSeq[v][j] and self.transferTo[v][j] is None:
                        raise RuntimeError('Request not dropped off at dropoff')

                    if req.D[r] == self.nodeSeq[v][j]:
                        dropOffTime[r] = self.tSeq[v][j]
                        if servedBy[r] != None:
                            raise RuntimeError('Request already dropped off by', v, j, servedBy[r])
                        servedBy[r] = v

                #Capacity check
                if (j == len(self.nodeSeq[v]) - 1 or self.nodeSeq[v][j + 1] != self.nodeSeq[v][j]) and np.sum(
                        req.groupSize[content]) > veh.capacity[v]:
                    raise RuntimeError('Vehicle capacity exceeded', v)

            #Vehicle not empty at its last node
            if len(content) > 0:
                raise RuntimeError('Requests still in the car', v)

            #Time decreased somewhere
            if np.any(np.diff(self.tSeq[v]) < 0):
                raise RuntimeError('Decreasing time!', v)

        if set(self.rejectedReqs) != set(np.where(~isPickedUp)[0]):
            raise RuntimeError('Unmatching failed requests', self.rejectedReqs, np.where(~isPickedUp)[0])

    #Post processes the solution to compute the route taken by the requests
    def postProcess(self):
        #Requests node sequence, time sequence
        self.rNodeSeq = [[req.P[i]] for i in range(req.n)]
        self.rTSeq = [[0.] for i in range(req.n)]
        #Vehicle transporting the request
        self.vehInSeq = [[None] for i in range(req.n)]
        #Number of transfers the request is involved in
        self.rNTransfers = np.zeros(req.n)

        #Parse the solution
        for v in range(len(self.nodeSeq)):
            for j in range(1, len(self.nodeSeq[v])):
                if self.gotIn[v][j] is not None:
                    r = self.gotIn[v][j]
                    self.rNodeSeq[r].append(self.nodeSeq[v][j])
                    self.rTSeq[r].append(self.tSeq[v][j]+0.0001)
                    self.vehInSeq[r].append(v)

                if self.gotOut[v][j] is not None:
                    r = self.gotOut[v][j]
                    self.rNodeSeq[r].append(self.nodeSeq[v][j])
                    self.rTSeq[r].append(self.tSeq[v][j])
                    self.vehInSeq[r].append(None)
                    if req.D[r] != self.nodeSeq[v][j]:
                        self.rNTransfers[r] += 1

        # Rearrange everything in correct order
        for i in range(req.n):
            #To numpy array
            self.rNodeSeq[i] = np.array(self.rNodeSeq[i])
            self.rTSeq[i] = np.array(self.rTSeq[i])
            self.vehInSeq[i] = np.array(self.vehInSeq[i])
            #Correct ordering
            order = np.argsort(self.rTSeq[i])
            self.rNodeSeq[i] = self.rNodeSeq[i][order]
            self.rTSeq[i] = self.rTSeq[i][order]
            self.vehInSeq[i] = self.vehInSeq[i][order]
            # if np.any(np.abs(np.diff(self.rTSeq[i])) < 0.0000001):
            #     raise RuntimeError('Ambiguous order',  np.where(np.abs(np.diff(self.rTSeq[i])) < 0.0000001)[0])
            self.rTSeq[i] = np.round(self.rTSeq[i])  # removes the 0.0001

    #Destroys the routes transporting request r
    def dropRequest(self, r):
        for i in range(self.nVeh):
            for j in range(len(self.gotIn[i])):
                if self.gotIn[i][j] == r:
                    self.destroyRoute(i, j)
                    break


    # Destroy everything after indDestroy (included) and propagates if necessary. Note that noDropoff does not propagate!
    def destroyRoute(self, vInd, indDestroy, noDropoff=False, nUnitsSpec=None):
        # Sometimes the destruction needs to be canceled => back up vars that __destroyRoute will modify
        if nUnitsSpec is not None:
            isHandled = req.isHandled.copy()
            cNode = veh.cNode.copy()
            cTime = veh.cTime.copy()
            isEmpty = veh.isEmpty.copy()

        #Copying routes is expensive and __destroyRoute at most overwrites them => don't modify directly the object
        certFeasibility = [None]*veh.n
        #Number of units the destroyed routes will have after the destruction
        nUnits = [len(i) for i in self.tSeq]

        #Compute the destruction
        self.__destroyRoute(vInd, indDestroy, nUnits, certFeasibility, noDropoff=noDropoff)

        #Check if we meet the nUnits specifications
        if nUnitsSpec is not None:
            for v, nMin in nUnitsSpec:
                #If we don't meet the specifications for nUnits
                if nUnits[v] <= nMin:
                    #Restore the changes to the solution
                    req.isHandled = isHandled
                    veh.cNode = cNode
                    veh.cTime = cTime
                    veh.isEmpty = isEmpty
                    #Stop and signal the failure
                    return False


        #Actually modify the solution
        for v in range(self.nVeh):
            #If the entire route of that v was modified
            if nUnits[v] != len(self.tSeq[v]):
                #Truncate the entire route
                self.truncate(v, nUnits[v])
                #Update the certificate of feasibility
                veh.certFeasibility[v] = certFeasibility[v]
                # To avoid getting stuck in the repair phase, we protect 50% of the routes we destroy
                self.canBeModified[v] &= np.random.rand() < .5
        #Destruction was succesful
        return True


    # Determines where every route should be destroyed without modify the solution object. However, it does modify veh and req!
    def __destroyRoute(self, vInd, indDestroy, nUnits, certFeasibility, noDropoff=False):
        # There is nothing to destroy
        if indDestroy >= nUnits[vInd]:
            return

        # We first destroy the route then propagate the destruction
        paramPropag = []

        # Now we move destroy the entire route unit by unit until we can prove with a certFeasibility that the entire route is feasibly completable
        j = nUnits[vInd]
        while j > 0:
            j -= 1

            # Remember to propagate the destruction to the vehicles affected by the destruction
            if self.transferTo[vInd][j] is not None:
                # Index of transfer for the other vehicle
                jOtherVeh = self.indTransfer(self.transferTo[vInd][j], vInd, self.nodeSeq[vInd][j], fromOtherVeh=True)
                if jOtherVeh is not None:
                    paramPropag.append([self.transferTo[vInd][j], jOtherVeh])
            if self.transferFrom[vInd][j] is not None:
                # The other vehicle might have reached the transfer before vInd => we need to find the time of the dropoff
                jOtherVeh = self.indTransfer(self.transferFrom[vInd][j], vInd, self.nodeSeq[vInd][j], fromOtherVeh=False)
                if jOtherVeh is not None:
                    paramPropag.append([self.transferFrom[vInd][j], jOtherVeh])

            # We are not serving this request anymore
            if self.gotIn[vInd][j] is not None and self.nodeSeq[vInd][j] == req.P[self.gotIn[vInd][j]]:
                req.isHandled[self.gotIn[vInd][j]] = False

            if j <= indDestroy:
                # Update the vehicle's state
                veh.cNode[vInd] = self.nodeSeq[vInd][j - 1]
                veh.cTime[vInd] = self.tSeq[vInd][j - 1]

                if noDropoff:
                    nUnits[vInd] = j
                    break

                # Occasionally we keep destroying for no particular reason
                if j > 1 and np.random.rand() < .25:
                    continue

                # If the vehicle is empty
                if self.reqInSeq[vInd][j - 1] == []:
                    # It might be not be the first time we are destroying this route and it might therefore have been assigned a dropoff route!
                    veh.isEmpty[vInd] = True
                    certFeasibility[vInd] = []
                    nUnits[vInd] = j
                    break

                # The vehicle is not empty; we try to find a certificate of feasibility for the vehicle
                ro = Route([], veh.cTime[vInd], veh.cNode[vInd], vInd, reqIn0=set(self.reqInSeq[vInd][j - 1]))
                if ro.compute(noPickup=True):
                    nUnits[vInd] = j
                    veh.certFeasibility[vInd] = ro
                    veh.isEmpty[vInd] = False
                    break
                # There is no certificate of feasibility, we have to keep destroying the solution

        # Propagate destruction.
        for p in paramPropag:
            self.destroyRoute(*p, nUnits, certFeasibility)

    #Destroy operator
    def destroy(self, noTransRepair, Q):
        # Reset pairedWith
        veh.pairedWith = np.full(veh.n, None)
        # Mark all request except the failed ones as handled
        req.isHandled |= True
        req.isHandled[self.rejectedReqs] = False
        #Get the performance of the entire routes and which entire routes are involved in a transfer
        vehPerf, inTransfer = self.vehPerf()
        self.canBeModified |= True
        # Protect some entire routes involved in a transfer against modification
        self.canBeModified[inTransfer] &= (np.random.rand((np.sum(inTransfer))) < .5 if not noTransRepair else False)


        # WORST REMOVAL: destroy worst performing routes
        # Decide how many routes to destroy
        nDestroy = np.random.randint(int(Q * 25) + 1) + 1
        if np.random.rand() < .75:
            vehPerf -= 100 * inTransfer
        #Select vehicles with poor performance
        routesToDestroy = np.argpartition(-vehPerf * np.random.uniform(1, 4, veh.n), nDestroy)[:nDestroy]
        for vInd in routesToDestroy:
            if not self.canBeModified[vInd]:
                continue
            # Destroy the route
            self.destroyRoute(vInd, 1, veh)

        #RANDOM REMOVAL: destroy a random route
        if np.random.rand() < .5:
            # Destroy a random route
            self.destroyRoute(np.random.randint(veh.n), 1, veh)

        # ZERO SPLIT REMOVAL: destroy routes at random zero split, right before last drop-off
        for v in sample(range(veh.n), np.random.randint(2 + int(5 * Q))):
            if not self.canBeModified[v]:
                continue
            #Parse the entire route, starting at a random position
            lenRo = len(self.nodeSeq[v])
            pos0 = np.random.randint(lenRo)
            for j in range(1, lenRo):
                pos = (pos0 + j) % (lenRo - 1) + 1
                # if zero split
                if len(self.reqIn[v][pos]) == 0 and len(self.reqIn[v][pos - 1]) > 0:
                    self.destroyRoute(v, pos, veh)
                    break

        # POST TRANSFER REMOVAL: destroy the dropoff route but not the transfer
        whereInTransfer, = np.where(inTransfer)
        for v in sample(list(whereInTransfer), k=min(2, len(whereInTransfer))):
            for j in range(len(self.nodeSeq[v]) - 1, 0, -1):
                if self.transferFrom[v][j] is not None or self.transferTo[v][j] is not None:
                    self.destroyRoute(v, j + 1, veh)
                    break

        # GREEDY RIDESHARING DESTROY OPERATOR: rebuild the entire route of a vehicle allowing the pick-up of any request (served or rejected).
        if np.random.rand() < 10 * Q ** 2:
            #Select the vehicle
            vg = routesToDestroy[np.random.randint(len(routesToDestroy))]
            #Create a route for vg serving one request
            tArrival = veh.cTime[vg] + tMat[veh.cNode[vg], req.P]
            feasibleArrival, = np.where((tArrival >= req.E - dat.maxWait) & (tArrival < req.ld))
            candR = list(feasibleArrival[tArrival[feasibleArrival] < (np.min(tArrival[feasibleArrival]) + 20)])
            rovg = Route([], self.tSeq[vg][0], self.nodeSeq[vg][0], vg)

            #Add requests to rovg
            rejected = set()
            while len(candR) > 0:
                r = np.random.randint(len(candR))
                rejected.add(candR[r])
                rovgcp = rovg.copy()
                if not rovgcp.insertRequest(candR.pop(r)):
                    continue
                # We accept the route
                rovg = rovgcp
                #Rebuild candR
                candR = list(set().union(*req.simRelated[rovg.reqIn]) - set(rovg.reqIn) - rejected)

            # Drop stolen requests
            for r in rovg.reqIn:
                self.dropRequest(r)

            if len(rovg.reqIn) > 0:
                # Log change in sol
                self.addRoute(rovg, vg)

        # RELATED ROUTE DESTROY OPERATOR: destroy routes related to unhandled requests
        if np.random.rand() < 20 * Q:
            notHandled = set(np.where(~req.isHandled)[0])
            routeScore = np.zeros(veh.n)
            #80% of the time, the score of a route is the number of requests it transports related a request not handled that
            if np.random.rand() < .8:
                for vInd in range(veh.n):
                    routeScore[vInd] = len(
                        notHandled & set().union(*req.simRelated[[i for i in self.gotIn[vInd] if i is not None]]))
            else:#score of the route is 1 if it can pickup an unhandled request
                for vInd in range(veh.n):
                    for rInd in notHandled:
                        if tMat[veh.originPos[vInd], req.P[rInd]] < req.ld[rInd]:
                            routeScore[vInd] = 1

            # Number of routes to destroy
            nDestroy = min(veh.n - 1, int(Q * len(notHandled)) + 2)
            for vInd in np.argpartition(sample(list(-routeScore), k=len(routeScore)), nDestroy)[:nDestroy]:
                if not self.canBeModified[vInd]:
                    continue
                # Destroy the route
                self.destroyRoute(vInd, 1, veh)

        #Make sure the requests inside a certificate of feasibility as handled
        for ro in veh.certFeasibility:
            if ro == []:
                continue
            # Mark the requests inside a vehicle as handled
            req.isHandled[ro.reqIn] = True

        # Force request insertion destroy mechanism
        notHandled = set(np.where(~req.isHandled)[0])
        for v in sample(range(veh.n), k=min(int(Q * veh.n), veh.n)):
            if not self.canBeModified[v]:
                continue
            # Do not consider vehicle if it is involved in transfer
            for tt, tf in zip(self.transferTo[v], self.transferFrom[v]):
                if tt is not None or tf is not None:
                    break
            else:
                if len(self.nodeSeq[v]) < 2:
                    continue
                reqIn = list(set().union(*self.reqIn[v]))
                candR = list(set().union(*req.fRelated[reqIn]) & notHandled)
                for r in self.sample(candR, k=len(candR)):
                    roCand = Route(reqIn + [r], self.time[v][0], self.nodeSeq[v][0], v)
                    if roCand.compute():
                        # print('SUCCESSFUL FORCED INSERTION', roCand.reqIn)
                        # Destroy the route
                        self.destroyRoute(v, 1, veh)
                        # Log change in newSol
                        self.addRoute(roCand, v, veh)
                        notHandled -= set(roCand.reqIn)
                        # print('Force request insertion destroy mechanism')
                        break

        #Reset rejectedReqs
        self.rejectedReqs = []


    #Truncates entire specified by vInd by removing every units after and including the unit with index j
    def truncate(self, vInd, j):
        self.gotIn[vInd] = self.gotIn[vInd][:j]
        self.gotOut[vInd] = self.gotOut[vInd][:j]
        self.nodeSeq[vInd] = self.nodeSeq[vInd][:j]
        self.tSeq[vInd] = self.tSeq[vInd][:j]
        self.isEmptySeq[vInd] = self.isEmptySeq[vInd][:j]
        self.transferTo[vInd] = self.transferTo[vInd][:j]
        self.transferFrom[vInd] = self.transferFrom[vInd][:j]
        self.reqInSeq[vInd] = self.reqInSeq[vInd][:j]

    # Finds the index at which vInd participated in a transfer with otherVeh at tNode
    def indTransfer(self, vInd, otherVeh, tNode, fromOtherVeh, nUnits):
        for i in range(nUnits[vInd]):
            # Transfer is from otherVeh
            if fromOtherVeh and self.transferFrom[vInd][i] == otherVeh and self.nodeSeq[vInd][i] == tNode:
                return i
            if (not fromOtherVeh) and self.transferTo[vInd][i] == otherVeh and self.nodeSeq[vInd][i] == tNode:
                return i

    def printSeq(self, i):
        return list(zip(self.nodeSeq[i], self.tSeq[i], self.gotIn[i], self.gotOut[i], self.transferFrom[i], self.transferTo[i], self.reqInSeq[i]))


