import scipy.io
import numpy as np
import random
import sys

def fuzzyKmeans(data, k):
    # Pick random points in the array as cluster centers
    centers = []
    nums = []
    count = 0
    while(count < k):
        num = random.randint(0, data.shape[0])
        if num not in nums:
            nums.append(num)
            centers.append(data[num])
            count += 1
            
    # Offset centers to avoid undefined weight
    for center in centers:
        center += 0.1
            
    # Convert the the list of arrays into an array of arrays
    centers = np.array(centers)
    
    # Create a list to hold distances between data point and certain cluster
    # Each row is a cluster center and each point in the row is a distance between point and cluster center
    distances = []
    for i in range(k):
        distances.append([])
    
    for center in range(k):
        for point in data:
            dist = np.linalg.norm(centers[center] - point)
            distances[center].append(dist)
            
    # Assign weights for each data point belonging to a certain cluster
    distances = np.array([np.array(x) for x in distances])
    sumOfDist = np.sum(distances, axis = 0)
    
    weights = []  
    for i in range(k):
        weights.append([])
        for point in range(data.shape[0]):
            if(distances[i][point] == 0):
                weights[i].append(sys.maxsize)
            else:
                weights[i].append(1 / (distances[i][point] / sumOfDist[point]))
    
    # Normalize weights
    weights = np.array([np.array(x) for x in weights])
    normalizeSums = np.sum(weights, axis = 0)
    
    for center in weights:
        for point in range(data.shape[0]):
            center[point] /= normalizeSums[point]
        
    # Update the center locations
    newCenters = np.zeros((k, data.shape[1]))
    centerSums = np.sum(weights, axis = 1)
    
    for i in range(k):
        for point in range(data.shape[0]):
            newCenters[i] += data[point] * weights[i][point]
        newCenters[i] /= centerSums[i]

def loadData(file, entry):
    data = scipy.io.loadmat(file)
    dList = data[entry]
    
    return dList

def reshape(a, b):
    # Reshapes a to have the same number of columns as b
    tempA = a
    while(tempA.shape[1] != b.shape[1]):
        tempA = np.delete(tempA, -1, axis = 1)
        
    return tempA
    
def main():
    # Load in matlab files and convert into numpy arrays
    green = loadData("green_data.mat", "green_data")
    red = loadData("red_data.mat", "red_data")
    
    # Compare the row length and match file with longer row to length of shorter row
    if(green.shape[1] > red.shape[1]):
        green = reshape(green, red)
    else:
        red = reshape(red, green)
    
    # Concatenate the green and red channel arrays
    combine = np.concatenate((green, red))
    
    fuzzyKmeans(combine, 3)
    
    
    

if __name__ == "__main__":
    main()