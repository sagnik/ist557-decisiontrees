import pickle
import numpy as np
import copy

def coauthwithsameconfjour(aid,pid,papauthdict,authconfidict,authjouridict):
    coauthors=copy.deepcopy(set(papauthdict[pid]))
    authorconfjourcor=[]
    orgauthorconfs=[]
    orgauthorjours=[]
    if int(aid) in coauthors: 
        coauthors.remove(int(aid))
    else:
        print "error"
        print coauthors
    if aid in authconfidict:
        orgauthorconfs=authconfidict[aid]
    if aid in authjouridict:
        orgauthorjours=authjouridict[aid]
    if len(orgauthorconfs)<1 and len(orgauthorjours)<1:
        return [0.0,0.0,0.0,0.0,0.0]
    for item in coauthors:
        thisauthorconfs=[]
        thisauthorjours=[]
    if str(item) in authconfidict:
        thisauthorconfs=authconfidict[str(item)]
    if str(item) in authjouridict:
        thisauthorjours=authjouridict[str(item)]
    i1=set(orgauthorconfs) & set(thisauthorconfs)
    i2=set(orgauthorjours) & set(thisauthorjours)
    authorconfjourcor.append(len(i1)+len(i2))
    authconfjourcorarr=np.asarray(authorconfjourcor)
    l=[]
    if authconfjourcorarr.any():
        l.append(np.mean(authconfjourcorarr))
        l.append(np.std(authconfjourcorarr))
        l.append(np.min(authconfjourcorarr))
        l.append(np.max(authconfjourcorarr))
        l.append(np.sum(authconfjourcorarr))
    else:
        l=[1.0,0.0,1.0,1.0,1.0]
    return l 

def main():
    print "features loading.."
    traindata=pickle.load(open("traineddata_org_features.list","r"))
    papauthdict=pickle.load(open("papauthidict.dict","r"))
    authconfidict=pickle.load(open("authconfidict.dict","r"))
    authjouridict=pickle.load(open("authjouridict.dict","r"))
    print "features loaded, extraction started"
    mfeatures=[]
    features = [x[2:] for x in traindata]
    for item in features:
        mfeatures.append(list(item))
    for index in range (0,len(traindata)):
        item=traindata[index]
    aid=item[0]
    pid=item[1]
    print str(aid)+" "+str(pid)
    x=coauthwithsameconfjour(str(aid),str(pid),papauthdict,authconfidict,authjouridict)
    for temps in x:
        mfeatures[index].append(temps)#index=5
        print str(index)+" out of "+str(len(traindata))+ " processed"
    print "writing training data"
    pickle.dump(mfeatures,open("traindata-allfeatures.list","w"))
    print "feature extraction finished \n"


if  __name__ =='__main__':
    main()

