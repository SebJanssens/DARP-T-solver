import numpy as np
from copy import deepcopy as cp
import LoadData as dat


# Class describing all the requests
class Requests:
    def __init__(self, earliest, latest, groupSize, pickup, dropoff):
        # Total number of requests
        self.n = len(earliest)
        # Earliest time of departure
        self.E = cp(earliest)
        # Latest time of departure
        self.L = cp(latest)
        # Request group size
        self.groupSize = cp(groupSize)
        # Pick-up node
        self.P = cp(pickup)
        # Drop-off node
        self.D = cp(dropoff)
        # Latest time of departure
        self.ld = self.L - dat.tMat[self.P, self.D]

        # Non constant variable
        self.isHandled = np.zeros(self.n, dtype=bool)

        # Pickup -> dropoff direction vector, using mdsLoc
        self.dirVec = dat.mdsLoc[self.D] - dat.mdsLoc[self.P]
        # Normalizes every row (so that dot product gives cos(alpha))
        self.dirVec = self.dirVec / np.linalg.norm(self.dirVec, axis=1)[:, None]
        # Matrix contaning the dot product of any two requests dirVec
        self.dotProd = self.dirVec @ self.dirVec.T
        self.similarDir = np.array([set(np.where(self.dotProd[i] > np.cos(70 / 57))[0]) for i in range(self.n)])

        #Sum of all the pickup->dropoff distances
        self.totTravelTime = np.sum(dat.tMat[self.P, self.D])

        # Compute requests relations
        self.preCompute()

    # Computes related, fRelated, symRelated and transCand
    def preCompute(self):
        # Request q is related to request r if the route starting at r.p serving r and q is shorter than d(r.p, r.d) + d(q.r, q.d) (see thesis report)
        # q pick-up time, qPickUp[i,j] = self.E[i] + cst.tMat[self.P[i], self.P[j]] (must be >= req.E[j] + maxWait)
        qPickUp = self.E[:, None] + dat.tMat[np.ix_(self.P, self.P)]  # rq
        # rqqr route
        qDropOff = qPickUp + dat.tMat[self.P, self.D]  # qq
        rDropOff = qDropOff + dat.tMat[np.ix_(self.D, self.D)].T  # qr
        # Feasible pick up of q (and feasible dropoff of r)
        fPickup = (self.ld >= qPickUp) & (qPickUp + self.ld[:, None] - self.E[:, None] >= self.E - dat.maxWait) & \
                  (rDropOff <= self.L[:, None]) & (rDropOff - self.E[:, None] <= .999 * (
                    dat.tMat[self.P, self.D] + dat.tMat[self.P, self.D][:, None]))
        # Feasible rqqr route
        feasible = fPickup & (qDropOff <= self.L)

        # rqrq route
        rDropOff = qPickUp + dat.tMat[np.ix_(self.P, self.D)].T  # qr
        qDropOff = rDropOff + dat.tMat[np.ix_(self.D, self.D)]  # rq
        fPickuprqrq = (self.ld >= qPickUp) & (qPickUp + self.ld[:, None] - self.E[:, None] >= self.E - dat.maxWait) & \
                      (rDropOff <= self.L[:, None]) & (qDropOff - self.E[:, None] <= .999 * (
                    dat.tMat[self.P, self.D] + dat.tMat[self.P, self.D][:, None]))
        fPickup |= fPickuprqrq
        feasible |= (fPickuprqrq & (qDropOff <= self.L))

        # rrrr route should not be considered
        np.fill_diagonal(fPickup, False)
        np.fill_diagonal(feasible, False)

        # q for which pickup is feasible
        self.related = np.array([set(np.where(fPickup[i, :])[0]) for i in range(self.n)])
        # q for which the route is feasible
        self.fRelated = np.array([set(np.where(feasible[i, :])[0]) for i in range(self.n)])
        # Symmetric version of related (r is symRelated to q <=> q is symRelated to r)
        self.symRelated = np.array(
            [self.related[i] | set([rInd for rInd in range(self.n) if i in self.related[rInd]]) for i in range(self.n)])

        # Precomputation for tranfers. Finds the q that might be involved in a transfer with r
        reqInsMat = (dat.tMat[self.P] + dat.tMat[:, self.D].T) / dat.tMat[self.P, self.D][:,
                                                         None]  # reqInsMat[r, q] = (tMat[req.P[r], q] +tMat[q,req.D[r]])/tMat[req.P[r], req.D[r]]
        tArrival = self.E[:, None] + dat.tMat[self.P]  # tArrival[r, q] = req.E[r] + tMat[req.P[r], q]
        tld = self.L[:, None] - dat.tMat[:, self.D].T  # tld[r, q] = req.L[r] - tMat[q, req.D[r]]
        # No more than 40% delay, 5 mins+ intersecting TW
        rInds, nInds = np.where((reqInsMat < 1.4) & (tArrival + 5 <= tld))
        self.transCand = np.array([set() for i in range(self.n)])
        for i in range(len(rInds)):
            for j in np.where(nInds == nInds[i])[0]:
                if i == j or not rInds[j] in self.similarDir[rInds[i]]:
                    continue
                if tld[rInds[i], nInds[i]] + 5 < tArrival[rInds[j], nInds[j]] or tld[rInds[j], nInds[j]] + 5 < tArrival[
                    rInds[i], nInds[i]]:
                    continue
                self.transCand[rInds[i]].add(rInds[j])


req = Requests(dat.earliest, dat.latest, dat.groupSize, dat.pickup, dat.dropoff)
