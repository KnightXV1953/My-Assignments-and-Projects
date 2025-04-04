#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#


#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes. 
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats
import random

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):
        """
        Input:
            

        """
        #print "   "        
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        #nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        try:
            nexamples, nfeatures=X.shape
        except:
            nexamples=1
        #---------End of Your Code-------------------------#
        #return score, Xlidx,Xridx
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#            
        
        #---------End of Your Code-------------------------#
        
    def information_gain(self, parent, left_values ,  right_values):    
        prob_left = len(left_values) / len(parent)
        prob_right = len(right_values) / len(parent)
        parent_impurity = self.impurity(parent)
        left_impurity = self.impurity(left_values)
        right_impurity = self.impurity(right_values)
        
        split_entropy =  prob_left*left_impurity+prob_right*right_impurity
        value = parent_impurity - split_entropy
        
        
        return value , split_entropy
        # info_gain = parent_impurity - (prob_left * left_impurity + prob_right * right_impurity)
        # return info_gain
        
    def impurity(self , y):
        values , counts = np.unique(y , return_counts=True)
        prob = counts / len(y)
        return -np.sum(prob * np.log2(prob))
        
        
    
    def evaluate_numerical_attribute(self,feat, Y):
        
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        
        #classes=np.unique(Y)
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # Same code as you written in DT assignment...
        sidx = np.argsort(feat)
        sorted_feat = feat[sidx]  # sorted feature values
        sorted_Y = Y[sidx]        # sorted labels corresponding to feature values
        
        min_entropy = 0
        best_gain = 0
        best_split = None
        Xlidx, Xridx = None, None
        
        for j in range(1 , len(sorted_feat)):
            if sorted_feat[j] == sorted_feat[j-1]:
                continue
            
            threshold = (sorted_feat[j] + sorted_feat[j-1]) / 2
            # Split data based on the threshold
            left_mask = sorted_feat < threshold
            right_mask = sorted_feat >= threshold

            left_Y = sorted_Y[left_mask]
            right_Y = sorted_Y[right_mask]

            # Calculate information gain
            info_gain , split_entropy = self.information_gain(Y, left_Y, right_Y)

            # Check if this split has the best gain so far
            if info_gain > best_gain:
                best_gain = info_gain
                best_split = threshold
                min_entropy = split_entropy
                Xlidx, Xridx = np.where(left_mask)[0], np.where(right_mask)[0]  # Indices of left and right splits

        return best_split, min_entropy, Xlidx, Xridx
        
        #---------End of Your Code-------------------------#
            
        #return split,mingain,Xlidx,Xridx

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...        
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        
        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        min_entropy = np.inf
        best_split, Xlidx, Xridx , best_score = None, None, None , None
        f_idx = 0
        random_feat = np.random.choice(nfeatures , self.nrandfeat , replace=False) 
        for j in random_feat:
            feat = X[:,j]
            split , entropy , lidx , ridx  = self.findBestRandomSplit(feat, Y)
            
            
            if len(lidx) == 0 or len(ridx) == 0:
                continue
        
            if entropy < min_entropy:
                min_entropy = entropy
                best_split = split
                Xlidx = lidx
                Xridx = ridx 
                f_idx = j
        return best_split , min_entropy , Xlidx , Xridx , f_idx
        #---------End of Your Code-------------------------#
        #return minscore, bXl,bXr

    def findBestRandomSplit(self,feat,Y):
        """
            
            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        #return splitvalue, score, np.array(indxLeft), indxRight

        #import pdb;         pdb.set_trace()
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        max_info_gain = -np.inf
        splitvalue = None
        indxLeft = None
        indxRight = None
        min_entropy = None

        # Randomly sample `nsplits` points from the feature range
        for i in range(self.nsplits):
            random_split = np.random.uniform(np.min(feat), np.max(feat))

            # Create masks for left and right splits
            left_mask = feat < random_split
            right_mask = feat >= random_split
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue  # Skip invalid splits

            # Split Y based on the masks
            Y_left = Y[left_mask]
            Y_right = Y[right_mask]

            # Calculate information gain
            info_gain , entropy = self.information_gain(Y, Y_left, Y_right)

            # Update the best split if the current split has higher information gain
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                min_entropy = entropy
                splitvalue = random_split
                indxLeft = np.where(left_mask)[0]
                indxRight = np.where(right_mask)[0]

        return splitvalue, min_entropy, indxLeft, indxRight
        
        #---------End of Your Code-------------------------#
        #return splitvalue, minscore, Xlidx, Xridx
    def calculateEntropy(self,Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which 
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """

        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        _, pl_counts = np.unique(lexam, return_counts=True)
        pl = pl_counts / float(len(lexam)) + np.spacing(1)

        _, pr_counts = np.unique(rexam, return_counts=True)
        pr = pr_counts / float(len(rexam)) + np.spacing(1)


        hl= -np.sum(pl*np.log2(pl)) 
        hr= -np.sum(pr*np.log2(pr)) 

        sentropy = pleft * hl + pright * hr

        return sentropy
    
    def evaluate(self , node , X):
        if X[node.fidx] <= node.split:
            return True
        else:
            return False



# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
        self.params = None
        self.a=0
        self.b=0
        self.c=0

        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        nexamples, nfeatures = X.shape
        min_entropy = np.inf
        best_params = None
        best_bias = None
        Xlidx, Xridx = None, None
        fidx = 0
        best_split = None

        for i in range(self.nsplits):

            params = np.random.normal(size=3)
            a , b , c = params
            decision =  (X[:,0] * a) + (X[:,1] * b) + c
            left_mask = decision <= 0
            right_mask = decision > 0
            
            Y_left = Y[left_mask]
            Y_right = Y[right_mask]

            
            info_gain, entropy = self.information_gain(Y, Y_left, Y_right)
            if entropy < min_entropy:
                min_entropy = entropy
                self.a = a
                self.b = b
                self.c = c
                best_split = decision
                Xlidx = np.where(left_mask)[0]
                Xridx = np.where(right_mask)[0]
                
        
        return best_split , min_entropy, Xlidx, Xridx , 0
        
        #---------End of Your Code-------------------------#
        return scores[minindx],xlind,xrind
        #return minscore, bXl, bXr
        
        

    def evaluate(self,wlearner_instance,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        decision =  (X[:,0] * self.a) + (X[:,1] * self.b) + self.c
        
        return decision <= 0
    
        #---------End of Your Code-------------------------#
        # build a classifier ax+by+c=0
class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D Conic based weak learner using 
        a random set of features from the given set of features...
    """
    def _init_(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters... 
        """
        RandomWeakLearner._init_(self,nsplits)
    
    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] training matrix...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        bfidx=-1, # best feature idx
        minscore=+np.inf
        
        tdim= np.ones((nexamples,1))# third dimension
        for i in np.arange(self.nsplits):
            # a*x^2+b*y^2+c*x*y+ d*x+e*y+f

            if i%5==0: # select features indeces after every five iterations
                fidx=np.random.randint(0,nfeatures,2) # sample two random features...
                # Randomly sample a, b and c and test for best parameters...
                parameters = np.random.normal(size=(6,1))
                # apply the line equation...
                res = np.dot ( np.hstack( (np.power(X[:,fidx],2),X[:,fidx],np.prod(X[:,fidx],1)[:,np.newaxis], tdim) ) , parameters )

            splits=np.random.normal(size=(2,1))
            
            # set split to -np.inf for 50% of the cases in the splits...
            if np.random.random(1) < 0.5:
                splits[0]=-np.inf

            tres=  np.logical_and(res >= splits[0], res < splits[1])
            
            score = self.calculateEntropy(Y,tres)

            if score < minscore:
                bfidx=fidx # best feature indeces
                bparameters=parameters # best parameters...
                minscore=score
                bres= tres
                bsplits=splits

        self.parameters=bparameters
        self.score=minscore
        self.splits=bsplits
        self.fidx=bfidx
        
        bXl=np.squeeze(bres)
        bXr=np.logical_not(bXl)

        return bsplits, minscore, bXl, bXr, fidx

    def evaluate(self,node,X):
        """
        Evalute the trained weak learner on the given example...
        """ 
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        fidx=self.fidx
        

        res = np.dot (np.hstack((np.power(X[:,fidx],2),X[:,fidx],np.prod(X[:,fidx],1)[:,np.newaxis], np.ones((X.shape[0],1)))), self.parameters) 
        return np.logical_and(res >= self.splits[0], res < self.splits[1])
    
"""    
wl=WeakLearner()
rwl=RandomWeakLearner()
lwl=LinearWeakLearner()
"""
