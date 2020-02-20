import numpy as np
import LoadData as dat
from requests import req
from vehicles import veh
from route import Route


#Given two segments described by [p[i], p[i] + d[i]], i=0,1, returns in y the two vectors (one in each segment) that are closest to one another
#See thesis report for the math behind the code
def segsegDist(p, d):
    def routine(i, y):
        j = int(not i)
        y[i] = p[i] if x[i] < 0 else p[i] + d[i]
        y[j] = p[j] + min(1, max(0, (y[i] @ d[j].T - d[j] @ p[j].T) / alpha[j])) * d[j]

    alpha = np.sum(d * d, axis=1)
    c = d @ (p[0] - p[1]) * [1, -1]
    beta = -d[0] @ d[1].T

    x = np.array([[alpha[1], -beta], [-beta, alpha[0]]]) @ c / (
            beta ** 2 - alpha[0] * alpha[1])
    y = p + x[:, None] * d

    if not 0 <= x[0] <= 1:
        routine(0, y)
        # At least one endpoint is getting involved => check other
        if not (0 <= x[1] <= 1):
            y2 = y.copy()
            routine(1, y2)
            # If closer
            dy, dy2 = y[1] - y[0], y2[1] - y2[0]
            if dy @ dy.T > dy2 @ dy2.T:
                y = y2
    elif not 0 <= x[1] <= 1:
        routine(1, y)

    return y

#Minimal distance between routes (lists of segments)
def roroDist(r1, r2, r1Time, r2Time):
    #Remove duplicates
    r1Ind = [i for i in range(len(r1) - 1) if r1[i] != r1[i + 1]] + [len(r1) - 1]
    r2Ind = [i for i in range(len(r2) - 1) if r2[i] != r2[i + 1]] + [len(r2) - 1]

    #If one of the routes is not long enough
    if len(r1Ind) < 2 or len(r2Ind) < 2:
        return dat.largeN, 0

    # Add time as extra dimension
    r1 = np.c_[dat.mdsLoc[np.array(r1)[r1Ind]], np.array(r1Time)[r1Ind]]
    r2 = np.c_[dat.mdsLoc[np.array(r2)[r2Ind]], np.array(r2Time)[r2Ind]]

    besty = []
    bestPos = []
    nBest = 1
    #Go through every pair of segments
    for i1 in range(len(r1) - 1):
        for i2 in range(len(r2) - 1):
            p = np.array([r1[i1], r2[i2]])
            d = np.array([r1[i1 + 1] - r1[i1], r2[i2 + 1] - r2[i2]])
            y = segsegDist(p, d)
            if len(besty) < nBest or ((y[1] - y[0]) @ (y[1] - y[0]).T < (besty[1] - besty[0]) @ (besty[1] - besty[0]).T):
                besty = y
                bestPos = [i1, i2]

    return np.linalg.norm(y[1] - y[0]), [r1Ind[bestPos[0]], r2Ind[bestPos[1]]]



#See cost function in thesis report
#Input is either one route, two routes or a transfer (2 pick up routes and two certificates of feasibility)
def cost(ro1=None, ro2=None, cf1=None, cf2=None):
    #Single route
    if ro2 is None:
        return (ro1.tSeq[-1] - ro1.t0)/ro1.sumf()
    if cf1 is None:
        return (ro1.tSeq[-1] - ro1.t0 + ro2.tSeq[-1] - ro2.t0)/(ro1.sumf() + ro2.sumf())
    return ((ro1.tSeq[-1] - ro1.t0 + ro2.tSeq[-1] - ro2.t0) + np.random.uniform(0.8, 1)*(cf1.tSeq[-1] - cf1.t0 + cf2.tSeq[-1] - cf2.t0))/(ro1.sumf() + ro2.sumf() + cf1.sumf() + cf2.sumf())


# There might be several insertion positions.
def reformatDuplicates(nodes, insPos):
    if len(nodes) == 0:
        return [], []
    prevIsDifferent = np.concatenate(([True], nodes[1:] != nodes[:-1]))
    #Reformatted nodes
    rfNodes = nodes[prevIsDifferent]
    #Reformatted insertion position
    rfInsPos = [[] for i in range(len(rfNodes))]
    rfInsPos[0].append(insPos[0])
    rfi = 0
    for i in range(1, len(prevIsDifferent)):
        if prevIsDifferent[i]:
            rfi += 1
        rfInsPos[rfi].append(insPos[i])
    return rfNodes, rfInsPos


# Returns the dropoff routes for a transfer between v1 and v2 happening at tNode
def getDropoffRoutes(v1, v2, tNode, tTransV1, tTransV2, tTransfer, reqV1, reqV2, allReq):
        setReqV1 = set(reqV1)
        setReqV2 = set(reqV2)

        # We will judge the quality of the partitioning based on
        # transfer -> dropoff direction vector, from mdsLoc
        dirVec = dat.mdsLoc[req.D[allReq]] - dat.mdsLoc[tNode]
        # Matrix contaning the dot product of any two requests dirVec /!\ Not normalized! /!\
        dotProd = dirVec @ dirVec.T

        # If the transfer happens where a request is picked up/dropped, it can not be exchanged!
        carLockedWInd = [i for i in range(len(reqV1)) if (tNode == req.P[reqV1[i]])]  # allReq indx
        carLockedViInd = [i + len(reqV1) for i in range(len(reqV2)) if (tNode == req.P[reqV2[i]])]  # allReq indx
        indToInd = [i for i in range(len(allReq)) if not (i in carLockedWInd or i in carLockedViInd)]

        # For all ways to split allReq in two identical vehicles
        for iV1, iV2 in twoPartition(len(indToInd)):
            # First to leave split swap should not happen when requests are car locked
            noSwap = (len(carLockedWInd) > 0) or (len(carLockedViInd) > 0)

            # To allReq indexing
            iV1 = [indToInd[i] for i in iV1] + carLockedWInd
            iV2 = [indToInd[i] for i in iV2] + carLockedViInd

            # Split-quality check, all reqs in a split should face the same direction
            if np.any(dotProd[np.ix_(iV1, iV1)] <= 0) or np.any(dotProd[np.ix_(iV2, iV2)] <= 0):
                continue

            # To req indexing
            splitV1 = allReq[iV1]
            splitV2 = allReq[iV2]
            # Group size of the splits
            gsSplitV1 = np.sum(req.groupSize[splitV1])
            gsSplitV2 = np.sum(req.groupSize[splitV2])

            # If capacity violation
            if gsSplitV1 > veh.capacity[v1] or gsSplitV2 > veh.capacity[v2]:
                # If swapping is not allowed or does not fix the capacity violation
                if noSwap or (gsSplitV2 > veh.capacity[v1] or gsSplitV1 > veh.capacity[v2]):
                    continue
                # Otherwise, we swap
                splitV1, splitV2 = splitV2, splitV1
                gsSplitV1, gsSplitV2 = gsSplitV2, gsSplitV1

            # Forbid an infeasible swap
            if (gsSplitV2 > veh.capacity[v1] or gsSplitV1 > veh.capacity[v2]):
                noSwap = True

            setSplitV1 = set(splitV1)
            setSplitV2 = set(splitV2)

            # Debug
            if len(setSplitV2) != len(splitV2) or len(setSplitV1) != len(splitV1):
                raise RuntimeError('Request present twice in split!')

            # If v1 and v2 have the same capacity and are simply swapping their requests and
            if veh.capacity[v1] == veh.capacity[v2] and setSplitV2 == setReqV1:
                continue

            # If no exchange of requests
            if setSplitV1 == setReqV1:
                continue

            # True if v1 must wait for v2 because v2 arrives at tNode after v1 and gives v1 a request
            v1MustWait = (not setSplitV1.issubset(setReqV1)) and tTransV1 < tTransV2
            # If swapping is allowed and v1 is waiting for v2 and swapping would allow v1 to drive off before tTransfer
            if not noSwap and v1MustWait and setSplitV2.issubset(setReqV1):
                v1MustWait = False
                splitV1, splitV2 = splitV2, splitV1
                setSplitV1, setSplitV2 = setSplitV2, setSplitV1
                # Don't swap twice!
                noSwap = True
            # Same for v2
            v2MustWait = not setSplitV2.issubset(setReqV2)
            if not noSwap and v2MustWait and (tTransV2 < tTransV1 and setSplitV1.issubset(setReqV2)):
                v2MustWait = False
                splitV1, splitV2 = splitV2, splitV1

            # Create dropoff routes
            dRoV1 = Route(splitV1, (tTransfer if v1MustWait else tTransV1), tNode, v1)
            dRoV2 = Route(splitV2, (tTransfer if v2MustWait else tTransV2), tNode, v1)

            # Compute best dropoff route and continue if there is none
            if (len(splitV1) > 0 and not dRoV1.compute(noPickup=True)) or (
                    len(splitV2) > 0 and (not dRoV2.compute(noPickup=True))):
                continue

            yield dRoV1, dRoV2

# See Stirling numbers of the second kind. Yields all S(n,2) pairs of subsets
def twoPartitionLegacy(n):
    yield ([], list(range(n)))

    # Yields all different ways of taking k elements from a set of n elements
    def pick(k, n, cPick):
        # No more elements to pick, yield
        if k == 0:
            yield cPick
            return
        # order of the k elements does not matter => WMA that they are sorted
        for i in range(cPick[-1] + 1 if len(cPick) > 0 else 0, n):
            # Copy because of Python's mutability
            cPickp = cPick[:]
            cPickp.append(i)
            # Pass on all yielded values
            yield from pick(k - 1, n, cPickp)

    # WMA that the first set is smaller or equal to the second set to
    # avoid repetition (e.g. 0|12 and 12|0) => we stop at k=n/2
    for k in range(1, int(n / 2) + 1):
        # If both sets have the same size, we force (wlog) 0 to belong to first set to avoid repetition
        if k == n / 2.:
            for set1 in pick(k - 1, n, [0]):
                set2 = [i for i in range(n) if i not in set1]
                yield (set1, set2)
        else:
            for set1 in pick(k, n, []):
                set2 = [i for i in range(n) if i not in set1]
                yield (set1, set2)


# Much simpler non-recursive implementation of twoPartitionLegacy. Similar output in a different order. This time, duplicates are avoided by always putting 0 in the first set
# Binary digits of a number determine the partitioning. Could be extended to S(n,x) by applying same principle to base x
def twoPartition(n):
    if n == 0:
        return ([], [])
    # For every number i that can be written with n-1 bits
    for i in range(2 ** (n - 1)):
        # 0 in set1
        set1, set2 = [0], []
        # To set1 (set2) is added the indices (starting at 1) of binary digits of i that are 1 (0)
        for j in range(n - 1):
            set1.append(j + 1) if (i >> j) & 1 else set2.append(j + 1)
        yield (set1, set2)



