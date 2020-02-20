import numpy as np
import LoadData as dat
from requests import req
from vehicles import veh


#Class describing a route
class Route:
    #Input: list of requests we want to pick up and transport (r), initial time (t0) and node (n0) of the route, vehicle capacity (vehCap)
    #boolean indicating if the route is incomplete (isIncomplete), requests already present at n0 (reqIn0)
    def __init__(self, r, t0, n0, v, isIncomplete=False, reqIn0=None):
        self.isIncomplete = isIncomplete
        #Is the shortest computed?
        self.isComputed = False

        # Starting time of the route
        self.t0 = t0
        # Starting node of the route
        self.n0 = n0
        #Associated vehicle
        self.v = v

        #Sequences
        self.tSeq = np.array([t0])
        self.nodeSeq = np.array([n0])

        # r should be a list
        r = list(r) if isinstance(r, np.ndarray) else (r if isinstance(r, list) else [r])

        # Requests inside at the starting node
        self.reqIn0 = set() if reqIn0 is None else set(reqIn0)
        # Requests transported
        self.reqIn = np.array(list(set(r).difference(self.reqIn0)) + list(self.reqIn0)).astype('int')
        #Booleans indicating which requests should be picked up
        self.toPickup = np.full(len(self.reqIn), True)
        # Requests already inside do not have to picked up
        if len(self.reqIn0) > 0:
            self.toPickup[-len(self.reqIn0):] = False
        # Booleans indicating which requests should be dropped off
        self.toDropoff = np.full(len(self.reqIn), True)


    # Much faster than deepcopy
    def copy(self):
        cpRo = Route(list(self.reqIn.copy()), self.t0, self.n0, self.v, self.isIncomplete, self.reqIn0.copy())
        cpRo.isComputed = self.isComputed
        cpRo.toPickup = self.toPickup.copy()
        cpRo.toDropoff = self.toDropoff.copy()
        if self.isComputed:
            cpRo.rSeq = self.rSeq.copy()
            cpRo.tSeq = self.tSeq.copy()
            cpRo.nodeSeq = self.nodeSeq.copy()
            cpRo.tSlackSeq = self.tSlackSeq.copy()
            cpRo.rInSeq = [i[:] for i in self.rInSeq]
        return cpRo


    def insertRequest(self, rInd, dropItOff=True, compute=True):
        self.reqIn = np.append(self.reqIn, rInd)
        self.toPickup = np.append(self.toPickup, True)
        self.toDropoff = np.append(self.toDropoff, dropItOff)
        if not dropItOff:
            self.isIncomplete = True

        if compute:
            return self.compute()


    # rSeq: current sequence of requests visits, t: current time, node: current position, ind: index of element in rSeq we are writing this recursion (happens to be recursion depth)
    # rIn: requests currently inside, pDone: pickups that are done, dDone: dropoffs that are done
    def __compute(self, rSeq, t, node, ind, rIn, pDone, dDone):
        # If some request that still needs to be picked up/dropped off can not be picked up/dropped off in time anymore
        if (not np.all(pDone) and t > np.max(req.ld[self.reqIn[~pDone]])) or (not np.all(dDone) and t > np.max(req.L[self.reqIn[~dDone]])):
            return

        # We visited all nodes, yay!
        if ind == self._totNodes:
            # If the route we found is shorter
            if self._tShortest > t:
                # Save found feasible solution
                self._rSeqShortest = rSeq
                self._tShortest = t
            return

        # Let us first consider visiting a pick up node
        for i in range(len(pDone)):
            # If pickup already done
            if pDone[i]:
                continue

            # Request we are considering
            r = self.reqIn[i]
            # Time at which pick up is reached
            tNew = t + dat.tMat[node, req.P[r]]
            # If pickup is feasible
            if (req.E[r] - dat.maxWait <= tNew <= req.ld[r]) and veh.capacity[self.v] >= np.sum(req.groupSize[rIn]) + \
                    req.groupSize[r]:
                # Correct ctp for if there's some waiting
                tNew = max(tNew, req.E[r])

                # Update current solution
                rSeqNew = rSeq.copy()
                rSeqNew[ind] = r
                pDoneNew = pDone.copy()
                pDoneNew[i] = True
                rInNew = rIn + [r]

                # Recurse
                self.__compute(rSeqNew, tNew, req.P[r], ind + 1, rInNew, pDoneNew, dDone)

        # Consider visiting a drop off node
        for i in range(len(dDone)):
            # If dropoff already done or pickup not done yet
            if dDone[i] or not pDone[i]:
                continue

            # Request we are considering
            r = self.reqIn[i]
            # Time at which pick up is reached
            tNew = t + dat.tMat[node, req.D[r]]
            # If feasible
            if tNew <= req.L[r]:
                # Update current solution
                rSeqNew = rSeq.copy()
                rSeqNew[ind] = r
                dDoneNew = dDone.copy()
                dDoneNew[i] = True
                rInNew = rIn[:]
                rInNew.remove(r)

                # Recurse
                self.__compute(rSeqNew, tNew, req.D[r], ind + 1, rInNew, pDone, dDoneNew)




    # Computes the shortest route serving the requests in self.reqIn
    # Pickups or dropoffs are skipped for elements of self.toPickup and self.toDropoff that are false
    def compute(self, noDropoff=False, noPickup=False):
        if noDropoff:
            self.toDropoff &= False
            self.isIncomplete = True
        if noPickup:
            self.toPickup &= False

        #Requests initially inside
        rIn0 = list(self.reqIn[~self.toPickup])
        #Stop if initial content exceeds capacity
        if np.sum(req.groupSize[rIn0]) > veh.capacity[self.v]:
            return False

        #Actual computation is done in the private method
        # Total number of nodes to visit (not including starting node)
        self._totNodes = np.sum(self.toPickup) + np.sum(self.toDropoff)
        # Shortest route is written by __compute here.
        self._rSeqShortest = []
        self._tShortest = np.inf

        #Actual computation
        self.__compute(np.zeros(self._totNodes).astype('int'), t=self.t0, node=self.n0, ind=0, rIn=rIn0, pDone=~self.toPickup, dDone=~self.toDropoff)
        #If no route is found
        if self._tShortest == np.inf:
            return False

        # Post processing
        # Different sequences describing the route. This time the sequences include the initial conditions

        #Sequence of indicating which request is being picked up/dropped off
        self.rSeq = np.append([int(dat.largeN)], self._rSeqShortest)
        #Node visiting sequence
        self.nodeSeq = np.zeros(self._totNodes+1).astype('int')
        self.nodeSeq[0] = self.n0
        #Time at which the nodes are reached
        self.tSeq = np.zeros(self._totNodes+1)
        self.tSeq[0] = self.t0
        #Requests inside sequence when leaving the node
        self.rInSeq = [rIn0]
        #Time slack at every node
        self.tSlackSeq = np.zeros(self._totNodes+1)

        #(Re)build the sequences
        for i in range(self._totNodes):
            #Request considered
            r = self.rSeq[i+1]
            isPickup = r not in self.rInSeq[i]
            self.nodeSeq[i+1] = req.P[r] if isPickup else req.D[r]
            self.tSeq[i+1] = self.tSeq[i] + dat.tMat[self.nodeSeq[i], self.nodeSeq[i+1]]
            self.rInSeq.append(self.rInSeq[-1][:])
            if isPickup:
                self.tSeq[i+1] = max(req.E[r], self.tSeq[i+1])
                self.rInSeq[-1].append(r)
            else:
                self.rInSeq[-1].remove(r)

        if self._tShortest != self.tSeq[-1]:
            raise RuntimeError('Failed to rebuild the time sequence')

        #Compute the time slack
        self.tSlackSeq[-1] = (req.L[self.rSeq[-1]] if any(self.toDropoff) else req.ld[self.rSeq[-1]]) - self.tSeq[-1]
        for i in range(1, self._totNodes)[::-1]:
            #Request considered
            r = self.rSeq[i]
            #If we picking up r
            if r in self.rInSeq[i]:
                self.tSlackSeq[i] = min(self.tSlackSeq[i + 1], req.ld[r] - self.tSeq[i])
            else:
                self.tSlackSeq[i] = min(self.tSlackSeq[i + 1], req.L[r] - self.tSeq[i])
        self.tSlackSeq[0] = self.tSlackSeq[1]

        self.isComputed = True
        return True

    def reqFlow(self, inFlow=True):
        if inFlow:
            return [r if r in rIn else None for r, rIn in zip(self.rSeq, self.rInSeq)]
        #Outflow
        return [None] + [None if r in rIn else r for r, rIn in zip(self.rSeq[1:], self.rInSeq[1:])]


    #As defined in the thesis report
    def sumf(self):
        #Building allReq, a dictionary {request: [nIn, nOut]}
        #Initial content
        allReq = {r: [self.n0] for r in self.rInSeq[0]}
        #Requests picked up and dropped-off during the route
        for r, n in zip(self.rSeq[1:], self.nodeSeq[1:]):
            if r in allReq:#Drop off
                allReq[r].append(n)
            else:#Pickup
                allReq.update({r: [n]})
        #Requests inside at the end
        for r in self.rInSeq[-1]:
            allReq[r].append(self.nodeSeq[-1])

        #Sum the f
        tot = 0
        for r in allReq:
            nIn, nOut = allReq[r]
            if nIn != req.P[r] and nOut != req.D[r]:
                tot += dat.tMat[nIn, req.D[r]] - dat.tMat[nOut, req.D[r]]
            else:
                tot += dat.tMat[req.P[r], req.D[r]] - dat.tMat[req.P[r], nIn] - dat.tMat[nOut, req.D[r]]

        return tot

    # Distance between the route and points
    # See thesis report for the math
    def distPoint2Route(self, points):
        # Remove duplicates from the node sequence
        route = self.nodeSeq[np.concatenate(([True], self.nodeSeq[1:] != self.nodeSeq[:-1]))]

        if len(route) < 2:
            return [[dat.largeN]]
        # Get coordinates
        points = dat.mdsLoc[points]
        route = dat.mdsLoc[route]

        # Segments a->b
        ba = route[1:] - route[:-1]  # b-a
        ba2 = ba / np.sum(ba * ba, axis=1)[:, None]  # b-a/||b-a||^2
        # Closest point to point[i] on line j is lam[i, j]*a[j] + (1-lam[i, j))*b[j]
        lam = np.maximum(0, np.minimum(1, np.sum(ba2 * (route[1:] - points[:, None]), axis=2)))
        # Coordinates of closest point
        d = route[1:] - (lam.T * np.repeat(ba[None, :, :], len(points), axis=0).T).T  # lol
        delta = np.swapaxes(d, 0, 1) - points
        dist2 = np.sum(delta * delta, axis=2).T  # Distance^2 matrix
        bestSeg = np.argmin(dist2, axis=1)

        # Returns distance, index of beginning of segment, closest point
        tmp = range(len(bestSeg))
        return np.sqrt(dist2[tmp, bestSeg]), bestSeg, d[tmp, bestSeg]


    # The element [i, j] of the returned matrix is the delay incurred by inserting node n[i] into the route between route[j] and route[j+1]
    # i.e. tMat[route[j], n[i]] + tMat[n[i], route[j+1]] - tMat[route[j], route[j+1]]
    def insertionMatrix(self, n):
        return dat.tMat[np.ix_(self.nodeSeq[:-1], n)].T + dat.tMat[np.ix_(n, self.nodeSeq[1:])] - dat.tMat[self.nodeSeq[:-1], self.nodeSeq[1:]]

    # Time delay caused by inserting n into route without otherwise modifying the route sequence
    # Return insertion delay and corresponding index (best insertion is between index and index+1 (if it exists))
    def insertionDelay(self, n):
        delays = dat.tMat[self.nodeSeq[:-1], n] + dat.tMat[n, self.nodeSeq[1:]] - dat.tMat[self.nodeSeq[:-1], self.nodeSeq[1:]]
        bestPos = np.argmin(delays)
        return (delays[bestPos], bestPos)