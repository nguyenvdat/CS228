###############################################################################
# factor graph data structure implementation 
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018
###############################################################################

from factors import *
import numpy as np

class FactorGraph:
    def __init__(self, numVar=0, numFactor=0):
        '''
        var list: index/names of variables
        domain list: the i-th element represents the domain of the i-th variable; 
                     for this programming assignments, all the domains are [0,1]
        varToFactor: list of lists, it has the same length as the number of variables. 
                     varToFactor[i] is a list of the indices of Factors that are connected to variable i
        factorToVar: list of lists, it has the same length as the number of factors. 
                     factorToVar[i] is a list of the indices of Variables that are connected to factor i
        factors: a list of Factors
        messagesVarToFactor: a dictionary to store the messages from variables to factors,
                            keys are (src, dst), values are the corresponding messages of type Factor
        messagesFactorToVar: a dictionary to store the messages from factors to variables,
                            keys are (src, dst), values are the corresponding messages of type Factor
        '''
        self.var = [None for _ in range(numVar)]
        self.domain = [[0,1] for _ in range(numVar)]
        self.varToFactor = [[] for _ in range(numVar)]
        self.factorToVar = [[] for _ in range(numFactor)]
        self.factors = []
        self.messagesVarToFactor = {}
        self.messagesFactorToVar = {}

    def evaluateWeight(self, assignment):
        '''
        param - assignment: the full assignment of all the variables
        return: the multiplication of all the factors' values for this assigments
        '''
        a = np.array(assignment, copy=False)
        output = 1.0
        for f in self.factors:
            output *= f.val.flat[assignment_to_indices([a[f.scope]], f.card)]
        return output[0]

    def getInMessage(self, src, dst, type="varToFactor"):
        '''
        param - src: the source factor/clique index
        param - dst: the destination factor/clique index
        param - type: type of messages. "varToFactor" is the messages from variables to factors; 
                    "factorToVar" is the message from factors to variables
        return: message from src to dst
        
        In this function, the message will be initialized as an all-one vector (normalized) if 
        it is not computed and used before. 
        '''
        if type == "varToFactor":
            if (src, dst) not in self.messagesVarToFactor:
                inMsg = Factor()
                inMsg.scope = [src]
                inMsg.card = [len(self.domain[src])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesVarToFactor[(src, dst)] = inMsg
            return self.messagesVarToFactor[(src, dst)]

        if type == "factorToVar":
            if (src, dst) not in self.messagesFactorToVar:
                inMsg = Factor()
                inMsg.scope = [dst]
                inMsg.card = [len(self.domain[dst])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesFactorToVar[(src, dst)] = inMsg
            return self.messagesFactorToVar[(src, dst)]


    def runParallelLoopyBP(self, iterations): 
        '''
        param - iterations: the number of iterations you do loopy BP
          
        In this method, you need to implement the loopy BP algorithm. The only values 
        you should update in this function are self.messagesVarToFactor and self.messagesFactorToVar. 
        
        Warning: Don't forget to normalize the message at each time. You may find the normalize
        method in Factor useful.
        '''      
        ###############################################################################
        # To do: your code here
        for i in range(iterations):
            # Pass message from var to factor
            print(i)
            for var in self.var:
                factors = self.varToFactor[var]
                for factor in factors:
                    other_factors = list(set(factors) - set([factor]))
                    # self.messagesVarToFactor[(var, factor)] = self.getInMessage(var, factor, type="varToFactor")
                    card=[len(self.domain[var])]
                    self.messagesVarToFactor[(var, factor)] = Factor(scope=[var], card=card, val = np.ones(card)/card[0])
                    for other_factor in other_factors:
                        that_msg = self.getInMessage(other_factor, var, type="factorToVar")
                        self.messagesVarToFactor[(var, factor)] = self.messagesVarToFactor[(var, factor)].multiply(that_msg)
                    self.messagesVarToFactor[(var, factor)] = self.messagesVarToFactor[(var, factor)].normalize()
            # Pass message from factor to var
            for factor in range(len(self.factors)):
                var_list = self.factorToVar[factor]
                # print(factor)
                for var in var_list:
                    other_vars = list(set(var_list) - set([var]))
                    self.messagesFactorToVar[(factor, var)] = self.factors[factor]
                    for other_var in other_vars:
                        that_msg = self.getInMessage(other_var, factor, "varToFactor")
                        self.messagesFactorToVar[(factor, var)] = self.messagesFactorToVar[(factor, var)].multiply(that_msg)
                    self.messagesFactorToVar[(factor, var)] = self.messagesFactorToVar[(factor, var)].marginalize_all_but([var]).normalize()

        ###############################################################################


    def estimateMarginalProbability(self, var):
        '''
        Estimate the marginal probabilities of a single variable after running 
        loopy belief propogation.  (This method assumes runParallelLoopyBP has
        been run)
        param - var: a single variable index
        return: numpy array of size 2 containing the marginal probabilities 
                that the variable takes the values 0 and 1
        
        example: 
        >>> factor_graph.estimateMarginalProbability(0)
        >>> [0.2, 0.8]
    
        Since in this assignment, we only care about the marginal 
        probability of a single variable, you only need to implement the marginal 
        query of a single variable.     
        '''
        ###############################################################################
        # To do: your code here  
        factor = self.varToFactor[var][0]
        calibrated_factor = self.factors[factor]
        for var in self.factorToVar[factor]:
            msg = self.getInMessage(var, factor, "varToFactor")
            calibrated_factor = calibrated_factor.multiply(msg)
        calibrated_factor = calibrated_factor.marginalize_all_but([var]).normalize()
        return calibrated_factor.val


        ###############################################################################


    def getMarginalMAP(self):
        '''
        In this method, the return value output should be the marginal MAP 
        assignments for the variables. You may utilize the method
        estimateMarginalProbability.
        
        example: (N=2, 2*N=4)
        >>> factor_graph.getMarginalMAP()
        >>> [0, 1, 0, 0]
        '''

        output = np.zeros(len(self.var))
        ###############################################################################
        # To do: your code here  
        for i, var in enumerate(self.var):
            marginal = self.estimateMarginalProbability(var)
            output[i] = np.argmax(marginal)


        ###############################################################################  
        return output