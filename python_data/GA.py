import pandas as pd
import numpy as np
import random
import copy


from BayesNet import *
from kFold import kFold
from BayesNet import BayesNet

structMap = {0:[1,2],1:[1,3],2:[1,4],3:[2,3],4:[2,4],5:[3,4]}   # Mapping of the structure position and the nodes that it connects


data = pd.read_csv('./iris.data',header=None)
data.columns = ['x1','x2','x3','x4','Class']
print("Data read")



label = data['Class']
data = data.drop('Class',axis = 1)

####   DISCRETIZING THE COLUMNS 
#### Equal Width Bins

# for col in data.columns:
#     data[col] = pd.cut(data[col],bins=3,labels=['0','1','2']).astype('int')   

#### Equal Frequency Bins
for col in data.columns:
    limits = list(data[col].quantile([0.0,0.33,0.67,1.0]))
    limits[3] = limits[3]+0.2
    
    data[col] = pd.cut(data[col],bins=limits,labels=['0','1','2'],right=False).astype('int')   

    
label[label=='Iris-setosa']=0
label[label=='Iris-versicolor']=1
label[label=='Iris-virginica']=2

data['Class'] = label.astype(int)

print("Preprocessing Done")


class Gene(object):
    
    def __init__(self,structure):
        self.structure = structure
        self.fitness = 0.0

def mutate(gene,pm):
    '''
    Randomly mutates segments of the structure based on the probability pm
    '''
    
    structure = copy.deepcopy(gene.structure)
    
    for i in range(len(structure)):
        
        if random.random()>pm:
            
            if structure[i]==0:
                structure[i] =random.choice([-1,1])
            elif structure[i] ==1:
                structure[i] =random.choice([-1,0])
            else:
                structure[i] =random.choice([1,0])
    
    mutatedGene = Gene(structure)
    return(mutatedGene)
                

def crossover(gene1,gene2):
    '''
    Computes a new structure based on the crossover of two parent structures
    '''
    structure1 = gene1.structure
    structure2 = gene2.structure
    
    
    crossoverGene = Gene(structure1[0:3]+structure2[3:6])
    return(crossoverGene)
    
    
    
    
def generatePop(n):
    '''
    Randomly generates population of size n
    '''
    
    
    popList = []
    
    for i in range(n):
        
        structure = []
        for pos in range(6):
            allele = random.choice([-1,0,1])
            structure.append(allele)
        
        popList.append(Gene(structure))
            
    return(popList)


def evaluateGene(data,gene):
    
    acc = kFold(data,structure=gene.structure,verbose=False)
    
    gene.fitness = acc
    return(acc)

def evaluatePop(data,pop):
    
    for i in range(len(pop)):
        acc = kFold(data,structure=pop[i].structure,verbose=False)
        pop[i].fitness = acc
        print("Gene : %d Fitness : %f"%(i,acc))
    
def runGA(data,popSize=5, pm = 0.7, crossoverLimit= 3, nIterations=3):
    
    #Generating a population
    pop = generatePop(popSize)
    
    evaluatePop(data,pop)
    pop = sorted(pop, key = lambda x : x.fitness, reverse=True)
    
    print("The best structure from raw data : %s Fitness : %f"%(pop[0].structure,pop[0].fitness))
    
    
    for iteration in range(nIterations):
        print("Iteration %d"%iteration)
        
        mutateList = []
        
        for i in pop:
            mutatedGene = mutate(i,pm)
            mutateList.append(mutatedGene)
            
        
        evaluatePop(data,mutateList)
        for x in mutateList:
            print("structure : %s fitness : %f"%(x.structure,x.fitness))
        
        crossoverList = []
        
        for parent1 in range(crossoverLimit-1):
            for parent2 in range(parent1+1,crossoverLimit):
                
                crossoverGene = crossover(pop[parent1],pop[parent2])
                
                crossoverList.append(crossoverGene)
            
        evaluatePop(data,crossoverList)
        for x in crossoverList:
            print("structure : %s fitness : %f"%(x.structure,x.fitness))
        
        pop = pop+mutateList+crossoverList
        pop = sorted(pop, key = lambda x : x.fitness, reverse=True)
        pop = pop[0:10]
    
    print("The best structure : %s Fitness : %f"%(pop[0].structure,pop[0].fitness))
    

runGA(data)