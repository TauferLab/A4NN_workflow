import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import shutil
from penguin.function_index import Parametric



def lowerBounds(numParameters,boundsString):
    bounds=boundsString.lower()
    boundList=[]

    if bounds == "none":
        for x in range(numParameters):
            boundList.append(np.NINF)
        return boundList

    else:
        bounds=bounds.split(",")
        for item in bounds:
            try:
                bound = float(item)
                boundList.append(bound)
            except:
                boundList.append(np.NINF)
        assert(len(boundList) == numParameters), "Specified number of lower bounds did not match number of parameters"
        return boundList

#function to take the string the user inputs for upper bounds and returns a list of the bounds to use
def upperBounds(numParameters,boundsString):
    bounds=boundsString.lower()
    boundList=[]

    if bounds == "none":
        for x in range(numParameters):
            boundList.append(np.inf)
        return boundList

    else:
        bounds=bounds.split(",")
        for item in bounds:
            try:
                bound = float(item)
                boundList.append(bound)
            except:
                boundList.append(np.inf)
        assert(len(boundList) == numParameters), "Specified number of upper bounds: "+str(len(boundList))+", did not match number of parameters: "+str(numParameters)
        return boundList

#return the function object matching a given string name
def get_parametric(function_name):
    p = Parametric()
    try:
        fn = getattr(p, function_name) 
    except:
        print("The function name you entered does not match a standard available parametric function.")
                    
    return fn

#takes a string of ints separated by "-" and returns an array of those ints.
def stringToArray(string, ext='-'):
    string=str(string)
    if string=="":
        return np.array([])

    array=np.array([])
    for i in string.split(ext):
        array=np.append(array,int(float(i)))
    return array

#find all DIRECTORIES containing non-hidden files ending in FILENAME
def getDataDirectories(DIRECTORY, FILENAME="valLoss.txt"):
    directories=[]
    for directory in os.scandir(DIRECTORY):
        for item in os.scandir(directory):
            if item.name.endswith(FILENAME) and not item.name.startswith("."):
                directories.append(directory.path)
    return directories

#get all non-hidden data files in DIRECTORY with extension EXT
def getDataFiles(DIRECTORY, EXT='txt'):
    datafiles=[]
    for item in os.scandir(DIRECTORY):
        if item.name.endswith("."+EXT) and not item.name.startswith("."):
            datafiles.append(item.path)
    datafiles.sort()
    return datafiles

#checking if loss ever doesn't decrease for numEpochs epochs in a row.
def stopsDecreasing(loss, epoch, numEpochs):
    minLoss=np.inf
    epochMin=0
    for i in range(0,loss.size):
        if loss[i] < minLoss:
            minLoss=loss[i]
            epochMin=epoch[i]
        elif (epoch[i]-epochMin) >= numEpochs:
            return i, minLoss
        
    return i, minLoss

#dirpath is where the accuracy and loss files are stored. want to move the files into the same format expected by grabNNData.
def createFolders(SEARCHDIR, SAVEDIR):
    for item in os.scandir(SEARCHDIR):
        name=str(item.name)
        files=name.split('-')
        SAVEFULLDIR=SAVEDIR+str(files[0])
        if not os.path.exists(SAVEFULLDIR):
            try:
                os.makedirs(SAVEFULLDIR)
            except FileExistsError:
                #directory already exists--must have been created between the if statement & our attempt at making directory
                pass
        shutil.move(item.path, SAVEFULLDIR+"/"+str(files[1]))
    

#a function to read in information (e.g. accuracy, loss) stored at FILENAME
def grabNNData(FILENAME, fitness_title, columns, header='infer', sep=' '):
    data = pd.read_csv(FILENAME, sep=sep, header=header)

    columns=columns.split(' ')
    columnData = ""

    fitness_name=str(fitness_title)
    try:
        assert ('epochs' in data.columns) and (fitness_name in data.columns)
        sortedData=data.sort_values(by="epochs", axis=0, ascending=True)
        epoch=np.array(sortedData['epochs'])
        fitness=np.array(sortedData[fitness_name])

        # if no additional columns to track...
        if columns[0] == "None":
            columnData = ""
        else:
            for item in columns:
                columnData+=str(np.array(sortedData[item])[0]) + ' '
        
        #stripping the trailing space off of columnData
        columnData=columnData[:-1]

        return epoch, fitness, columnData

    except:
        print("Missing a column in NN datafile. Check for epochs and {}.".format(fitness_name))
        raise Exception('NN datafile is missing one of the expected columns: epochs or ' + fitness_name)


#slice data could be used to test values of E other than E=0.5, which we use by default
def sliceData(xsize, x, y, z=None, w=None):
    #we can slice the data to sample less often, but not more often. We verify that we're not being asked for a granularity that is smaller than the frequency of datapoints in the vectors.
    if x[0] > xsize:
        return x,y,z,w
    else:
        result=(1.0/x[0])*xsize
        #result is how often we should take datapoints if we wish to consider values every xsize

        x=x[int(result-1)::int(result)] 
        y=y[int(result-1)::int(result)]

        if z is not None:
            z=z[int(result-1)::int(result)]
            if w is None:
                return x,y,z
        else:
            return x,y

        #if we get to this point in function, it means z and w are both not None.
        w=w[int(result-1)::int(result)]
        return x,y,z,w