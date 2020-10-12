import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.special import gammaln, digamma, logsumexp
from autograd.scipy.special import logsumexp

from ssm.util import random_rotation, ensure_args_are_lists, \
    logistic, logit, one_hot
from ssm.regression import fit_linear_regression
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
import ssm.stats as stats


from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.cluster import SpectralClustering, AgglomerativeClustering


def sse(yact,ypred):
    return np.sum((yact-ypred)**2)


def observations_init_func_extra(self,datas,**kwargs):

    if 'init' in kwargs:
        init = kwargs['init']
    else:
        init = 'rand' #Default

    # Sample time bins for each discrete state.
    # Use the data to cluster the time bins if specified.
    K, D, M, lags = self.K, self.D, self.M, self.lags
    Ts = [data.shape[0] for data in datas]

    #Get size of windows and gap between start of windows for some methods (in units of time bins)
    if init=='window' or init=='ar_clust':
        if 't_win' in kwargs:
            t_win = kwargs['t_win']
        else:
            t_win = 10 #default
        if 't_gap' in kwargs:
            t_gap = kwargs['t_gap']
        else:
            t_gap = int(np.ceil(t_win/3)) #default
        # print('t_win:', t_win)
        # print('t_gap:', t_gap)

    #KMeans clustering
    if init=='kmeans':
        km = KMeans(self.K)
        km.fit(np.vstack(datas))
        zs = np.split(km.labels_, np.cumsum(Ts)[:-1])

    #Random assignment
    elif init=='rand' or init =='random': #Random
        zs = [npr.choice(self.K, size=T) for T in Ts]


    #Fits dynamics matrix on sliding segments of data, and see how well those dynamics fit other segments - then cluster this matrix of model errors
    elif init=='ar_clust':

        num_trials=len(datas)
        segs=[] #Elements of segs contain triplets of 1) trial, 2) time point of beginning of segment, 3) time point of end of segment

        #Get all segments based on predefined t_win and t_gap
        for tr in range(num_trials):
            T=Ts[tr]
            n_steps=int((T-t_win)/t_gap)+1
            for k in range(n_steps):
                segs.append([tr,k*t_gap,k*t_gap+t_win])

        #Fit a regression (solve for the dynamics matrix) within each segment
        num_segs=len(segs)
        sse_mat=np.zeros([num_segs,num_segs])
        for j,seg in enumerate(segs):
            [tr,t_st,t_end]=seg
            X=datas[tr][t_st:t_end+1,:]
            rr=Ridge(alpha=1,fit_intercept=True)
            rr.fit(X[:-1],X[1:]-X[:-1])

            #Then see how well the dynamics from segment J works at making predictions on segment K (determined via sum squared error of predictions)
            for k,seg2 in enumerate(segs):
                [tr,t_st,t_end]=seg2
                X=datas[tr][t_st:t_end+1,:]
                sse_mat[j,k]=sse(X[1:]-X[:-1],rr.predict(X[:-1]))

        #Make "sse_mat" into a proper, symmetric distance matrix for clustering
        tmp=sse_mat-np.diag(sse_mat)
        dist_mat=tmp+tmp.T

        #Cluster!
        clustering=SpectralClustering(n_clusters=self.K,affinity='precomputed').fit(1/(1+dist_mat/t_win))
        # clustering = AgglomerativeClustering(n_clusters=K,affinity='precomputed',linkage='average').fit(dist_mat/t_win)


        #Now take the clustered segments, and use them to determine the cluster of the individual time points
        #In the scenario where the segments are nonoverlapping, then we can simply assign the time point cluster as its segment cluster
        #In the scenario where the segments are overlapping, we will let a time point's cluster be the cluster to which the majority of its segments belonged
        #Below zs_init is the assigned discrete states of each time point for a trial. zs_init2 tracks the clusters of each time point across all the segments it's part of

        zs=[]
        for tr in range(num_trials):
            xhat=datas[tr]
            T=xhat.shape[0]
            n_steps=int((T-t_win)/t_gap)+1
            t_st=0
            zs_init=np.zeros(T)
            zs_init2=np.zeros([T,K]) #For each time point, tracks how many segments it's part of belong to each cluster
            for k in range(n_steps):
                t_end=t_st+t_win
                t_idx=np.arange(t_st,t_end)
                if t_gap==t_win:
                    zs_init[t_idx]=clustering.labels_[k]
                else:
                    zs_init2[t_idx,clustering.labels_[k]]+=1
                t_st=t_st+t_gap
            if t_gap!=t_win:
                max_els=zs_init2.max(axis=1)
                for t in range(T):
                    if np.sum(zs_init2[t]==max_els[t])==1:
                        zs_init[t]=np.where(zs_init2[t]==max_els[t])[0]
                    else:
                        if zs_init[t-1] in np.where(zs_init2[t]==max_els[t])[0]:
                            zs_init[t]=zs_init[t-1]
                        else:
                            zs_init[t]=np.where(zs_init2[t]==max_els[t])[0][0]

            zs.append(np.hstack([0,zs_init[:-1]])) #I think this offset is correct rather than just using zs_init, but it should be double checked.

    #Cluster based on the means and mean absolute difference of segments
    elif init=='window':

        num_trials=len(datas)
        n_steps_all=[]

        # Get values to cluster
        vals=[]
        for tr in range(num_trials):
            t_st=0
            T=Ts[tr]
            xhat=datas[tr]
            n_steps=int((T-t_win)/t_gap)+1
            n_steps_all.append(n_steps)

            for k in range(n_steps):
                if k==n_steps-1:
                    t_end=T-1
                else:
                    t_end=t_st+t_win
                t_idx=np.arange(t_st,t_end)
                X1=xhat[t_st:t_end,:]
                X2=xhat[t_st+1:t_end+1,:]


                #CLUSTER BY DIFFS
                tmp=np.mean(np.abs(X2-X1),axis=0) #mean absolute difference within segment
                tmp2=np.mean(X1,axis=0) #mean value within segement
                vals.append(np.hstack([tmp,tmp2])) #concatenate the above for clustering
                # As.append(np.mean(np.abs(X2-X1),axis=0))

                t_st=t_st+t_gap
        vals_all=np.vstack(vals) # combine across all trials


        ## Cluster ##
        km=KMeans(n_clusters=K)
        km.fit(vals_all)


        #Now take the clustered segments, and use them to determine the cluster of the individual time points
        #In the scenario where the segments are nonoverlapping, then we can simply assign the time point cluster as its segment cluster
        #In the scenario where the segments are overlapping, we will let a time point's cluster be the cluster to which the majority of its segments belonged
        #Below zs_init is the assigned discrete states of each time point for a trial. zs_init2 tracks the clusters of each time point across all the segments it's part of
        tr_st_idxs=np.hstack([0, np.cumsum(n_steps_all)])
        zs=[]
        for tr in range(num_trials):
            xhat=datas[tr]
            T=xhat.shape[0]
            n_steps=int((T-t_win)/t_gap)+1
            t_st=0
            zs_init=np.zeros(T)
            zs_init2=np.zeros([T,K])
            for k in range(n_steps):
                t_end=t_st+t_win
                t_idx=np.arange(t_st,t_end)
                if t_gap==t_win:
                    zs_init[t_idx]=km.labels_[k]
                else:
                    zs_init2[t_idx,km.labels_[k]]+=1
                t_st=t_st+t_gap
            if t_gap!=t_win:
    #             zs_init=np.argmax(zs_init2,axis=1)

                max_els=zs_init2.max(axis=1)
                for t in range(T):
                    if np.sum(zs_init2[t]==max_els[t])==1:
                        zs_init[t]=np.where(zs_init2[t]==max_els[t])[0]
                    else:
                        if zs_init[t-1] in np.where(zs_init2[t]==max_els[t])[0]:
                            zs_init[t]=zs_init[t-1]
                        else:
                            zs_init[t]=np.where(zs_init2[t]==max_els[t])[0][0]



            zs.append(np.hstack([0,zs_init[:-1]]))
            self.zs_init=zs_init



    else:
        print('Not an accepted initialization type')


    return zs
