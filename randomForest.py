#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

#---------------Instructions------------------#
# Please read the function documentation before
# proceeding with code writing. 

# For randomizing, you will need to use following functions
# please refer to their documentation for further help.
# 1. np.random.randint
# 2. np.random.random
# 3. np.random.shuffle
# 4. np.random.normal 


# Other Helpful functions: np.atleast_2d, np.squeeze()
# scipy.stats.mode, np.newaxis

#-----------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#
import tree as tree
import numpy as np
import scipy.stats as stats



class RandomForest:
    ''' Implements the Random Forest For Classification... '''
    def __init__(self, ntrees=10,treedepth=5,usebagging=False,baggingfraction=0.6,
        weaklearner="Conic",
        nsplits=10,        
        nfeattest=None, posteriorprob=False,scalefeat=True ):        
        """      
            Build a random forest classification forest....

            Input:
            ---------------
                ntrees: number of trees in random forest
                treedepth: depth of each tree 
                usebagging: to use bagging for training multiple trees
                baggingfraction: what fraction of training set to use for building each tree,
                weaklearner: which weaklearner to use at each interal node, e.g. "Conic, Linear, Axis-Aligned, Axis-Aligned-Random",
                nsplits: number of splits to test during each feature selection round for finding best IG,                
                nfeattest: number of features to test for random Axis-Aligned weaklearner
                posteriorprob: return the posteriorprob class prob 
                scalefeat: wheter to scale features or not...
        """

        self.ntrees=ntrees
        self.treedepth=treedepth
        self.usebagging=usebagging
        self.baggingfraction=baggingfraction

        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.nfeattest=nfeattest
        
        self.posteriorprob=posteriorprob
        
        self.scalefeat=scalefeat
        self.max = 0
        self.min = 0
        
        pass    
    def findScalingParameters(self,X):
        """
            find the scaling parameters
            input:
            -----------------
                X= m x d training data matrix...
        """
        self.max = np.max(X , axis=0)
        self.min = np.min(X , axis=0)
    
        
    def applyScaling(self,X):
        """
            Apply the scaling on the given training parameters
            Input:
            -----------------
                X: m x d training data matrix...
            Returns:
            -----------------
                X: scaled version of X
        """
        #return X
        self.range = self.max - self.min
        
        return (X-self.min) / self.range
        
    def train(self,X,Y,vX=None,vY=None):
            '''
            Trains a RandomForest using the provided training set..
            
            Input:
            ---------
            X: a m x d matrix of training data...
            Y: labels (m x 1) label matrix

            vX: a n x d matrix of validation data (will be used to stop growing the RF)...
            vY: labels (n x 1) label matrix

            Returns:
            -----------

            '''
            import tools as t

            nexamples, nfeatures= X.shape

            self.findScalingParameters(X)
            if self.scalefeat:
                X=self.applyScaling(X)

            self.trees=[]
            
            #-----------------------TODO-----------------------#
            #--------Write Your Code Here ---------------------#
            for i in range(self.ntrees):
                randomindexes = np.random.choice(nexamples , int(self.baggingfraction * nexamples) , replace=True)
                X_train = X[randomindexes]
                Y_train = Y[randomindexes]
                
                tree_instance = tree.DecisionTree(purity = 0.82 , exthreshold = 5 , maxdepth = self.treedepth , weaklearner=self.weaklearner, nsplits=self.nsplits, nfeattest=self.nfeattest )
                tree_instance.train(X_train , Y_train)
                self.trees.append(tree_instance)
                        
            #---------End of Your Code-------------------------#
        
    def predict(self, X):
        
        """
        Test the trained RF on the given set of examples X
        
                   
            Input:
            ------
                X: [m x d] a d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        z=[]
        if self.scalefeat:  
            X=self.applyScaling(X)

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        predictions = []
        for tree in self.trees:
            z.append(tree.predict(X))

        # Use majority voting
        arr = np.array(z)
        final_predictions = stats.mode(arr , axis = 0)
        majority = np.ravel(final_predictions.mode)  # Get the mode for each sample
        return majority
        
        
        #---------End of Your Code-------------------------#
        
