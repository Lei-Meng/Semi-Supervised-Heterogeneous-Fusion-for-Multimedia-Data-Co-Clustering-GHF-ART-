
import scipy.io as io
import numpy as np

def GHFART(M,N,root_path):
    '''
    % M: numpy arrary; m*n1 matrix of visual features; m is number of objects and n1 is number of visual features
    % N: numpy array; m*n2 matrix of textual features; n is the number of words
    '''
#-----------------------------------------------------------------------------------------------------------------------
# Input parameters
    alpha = 0.01 # no need to tune; used in choice function; to avoid too small cluster weights (resulted by the learning method of ART; should be addressed sometime); give priority to choosing denser clusters
    beta = 0.6 # has no significant impact on performance with a moderate value of [0.4,0.7]

    # the two rhos need carefully tune; used to shape the inter-cluster similarity; rho_v = 0.7 indicates an object will not be clustered to a cluster with visual similarity lower than 0.7
    rho_v = 0.6
    rho_t = 0.05

    gamma = 0.5 #note that gamma will be self-adapted and no need to tune; it is the weights for fusing visual and textual similarities
# Input parameters
#-----------------------------------------------------------------------------------------------------------------------

    #complement coding
    M = np.concatenate([M,1-M], 1)
    #Note that no complement coding for textual features N

    #get data sizes
    row, colV = M.shape
    _, colT = N.shape #note row, i.e. the number of objects, is the same for the two matrices

    #set clusters
    Wv = np.zeros((row, colV))
    Wt = np.zeros((row, colT))
    J = 0 # number of clusters
    L = np.zeros(row) # note the maximun number of cluster is row; record the sizes of clusters for the learning of textual features

    #record cluster assignment of objects in database
    Assign = np.zeros(row,dtype=np.int) # the cluster assignment of objects


    # for update of gamma
    # note that these are also the initialization
    Difference_V = np.zeros(row)
    Difference_T = np.zeros(row)
    AvgDif_V = 0
    AvgDif_T = 0
    R_V = 1 # robustness of  visual features
    R_T = 1 # robustness of  textual features

# -----------------------------------------------------------------------------------------------------------------------
# Clustering process

    print("algorithm starts")
    #first cluster
    Wv[0, :] = M[0, :]
    Wt[0, :] = N[0, :]

    J = 1
    Assign[0] = J-1 #note that python array index trickily starts from 0
    L[J-1] = 1

    #processing other objects
    for n in range(1,row):

        print('Processing image %d' % n)

        T_max = -1 #the maximun choice value
        winner = -1 #index of the winner cluster

        #compute the similarity with all clusters; find the best-matching cluster
        for j in range(0,J):

            #compute the match function
            Mj_numerator_V = 0
            Mj_numerator_T = 0

            for i in range(0,colV):
                Mj_numerator_V = Mj_numerator_V + min(M[n, i], Wv[j, i])

            for i in range(0,colT):
                Mj_numerator_T = Mj_numerator_T + min(N[n, i], Wt[j, i])

            Mj_V = Mj_numerator_V / sum(M[n,:])
            Mj_T = Mj_numerator_T / (0.00001 + sum(N[n, :])) #note that the addition of 0.00001 is to avoid the case when all words of an object are filtered

            if Mj_V >= rho_v and Mj_T >= rho_t:
                #compute choice function
                Tj = (1-gamma) * Mj_numerator_V / (alpha + sum(Wv[j, :])) + gamma * Mj_numerator_T / (alpha + sum(Wt[j,:]))

                if Tj > T_max:
                    T_max = Tj
                    winner = j


        #Cluster assignment process
        if winner == -1: #indicates no cluster passes the vigilance parameter - the rhos
            #create a new cluster
            J = J + 1
            Wv[J - 1, :] = M[n, :]
            Wt[J - 1, :] = N[n, :]
            Assign[n] = J - 1
            L[J - 1] = 1

            #update vigilance parameter gamma
            gamma = ((R_T) ** (J / (J + 1))) / ((R_T) ** (J / (J + 1)) + (R_V) ** (J / (J + 1))) # gamma is for textual features and that for visual feature is 1-gamma

        else: #if winner is found, do cluster assignment and update cluster weights and gamma

            #variables for computing gamma
            Wv_old = Wv[winner, :]
            Wt_old = Wt[winner, :]
            Pattern_V = M[n, :]
            Pattern_T = N[n, :]

            #update cluster weights
            for i in range(0, colV):
                Wv[winner, i] = beta * min(Wv[winner, i], M[n, i]) + (1 - beta) * Wv[winner, i]

            for i in range(0, colT):
                Wt[winner, i] = L[winner] / (L[winner] + 1) * (Wt[winner, i] + N[n, i] / L[winner])

            #cluster assignment
            Assign[n] = winner
            L[winner] += 1

            #update gamma
            Wv_now = Wv[winner, :]
            Wt_now = Wt[winner, :]

            NewDif_V = (L[winner] - 1) / L[winner] / sum(Wv_now) * (sum(Wv_old) * Difference_V[winner] + sum(abs(Wv_old - Wv_now)) + 1 / (L[winner]-1) * sum(abs(Wv_now-Pattern_V)))
            AvgDif_V = AvgDif_V + (NewDif_V - Difference_V[winner]) / J
            Difference_V[winner] = NewDif_V
            R_V = np.exp(-AvgDif_V)

            NewDif_T = (L[winner] - 1) / L[winner] / sum(Wt_now) * (sum(Wt_old) * Difference_T[winner] + sum(abs(Wt_now - ((L[winner] - 1) / L[winner]) * Wt_old)) + 1 / (L[winner] - 1) * sum(abs(Wt_now - Pattern_T)))
            AvgDif_T = AvgDif_T + (NewDif_T - Difference_T[winner]) / J
            Difference_T[winner] = NewDif_T
            R_T = np.exp(-AvgDif_T)

            gamma = R_T / (R_V + R_T)

# Clustering process
# -----------------------------------------------------------------------------------------------------------------------


    print("algorithm ends")

    #Clean indexing data
    colV = colV // 2
    Wv = Wv[0 : J, 0 : colV]
    Wt = Wt[0 : J, :]
    L = L[0 : J]

    
    # Store indexing base structure files

    io.savemat(root_path + 'J.mat', {'J': J})
    io.savemat(root_path + 'Wv.mat', {'Wv': Wv})
    io.savemat(root_path + 'Wt.mat', {'Wt': Wt})
    io.savemat(root_path + 'L.mat', {'L': L})
  


    return 0

