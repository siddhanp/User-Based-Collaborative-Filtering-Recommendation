import os
import sys
import itertools
import copy
import math
import time

# START TIME.
start_time = time.time()

os.environ['SPARK_HOME']="/Users/USC/Desktop/spark-1.6.1-bin-hadoop2.4"
sys.path.append("/Users/USC/Desktop/spark-1.6.1-bin-hadoop2.4/python")

from pyspark import SparkContext
sc = SparkContext()

# TAKE INPUT FROM COMMAND LINE.
path = os.path.join(sys.argv[1])
train = sc.textFile(path)
train = train.map(lambda x: ((x.split(",")[0], x.split(",")[1]), x.split(",")[2])).collect()
train = train[1:]
path2 = os.path.join(sys.argv[2])
test = sc.textFile(path2)
test = test.map(lambda x: ((x.split(",")[0], x.split(",")[1]), 0)).collect()
test = test[1:]
train = sc.parallelize(train)
train2 = train.map(lambda x: ((int(x[0][0]), int(x[0][1])), float(x[1]))).collect() 

# REMOVE TEST DATA FROM TRAINING DATA.
test = sc.parallelize(test)
train = sorted(train.subtractByKey(test).collect())

# GET AVERAGE PER USERID FOR PREDICTING MISSING VALUES.
avgPerUserID = {}
train_avg = sc.parallelize(train).map(lambda x: (int(x[0][0]), float(x[1]))).groupByKey().mapValues(list).map(lambda x: (x[0], sum(x[1])/len(x[1]))).collect()
for i in range(len(train_avg)):
    avgPerUserID[train_avg[i][0]] = train_avg[i][1]

# FORM USER-ITEM MATRIX AND USERS GROUPED-BY ITEM.
userItemMatrix, usersByItem = {}, {}
for i in range(len(train)):

    if int(train[i][0][1]) in usersByItem:
        usersByItem[int(train[i][0][1])].append(int(train[i][0][0]))
    else:
        usersByItem[int(train[i][0][1])] = [int(train[i][0][0])]      
    
    if int(train[i][0][0]) not in userItemMatrix:
        userItemMatrix[int(train[i][0][0])] = {}
    userItemMatrix[int(train[i][0][0])][int(train[i][0][1])] = float(train[i][1])

# FORM LIST OF USERS.
users = []
for i in userItemMatrix.iterkeys():
    users.append(i)

# CALCULATE PEARSON CORRELATION COEFFICIENT.
weight, co_rated = {}, {}
for x in itertools.combinations(users, 2):
    
    u1, u2 = x[0], x[1]
    sum1, sum2 = 0, 0
    x = tuple(sorted(x))
    for i in userItemMatrix[u1].iterkeys():
        if i in userItemMatrix[u2]:
            if x in co_rated:
                co_rated[x].append(i)
            else:
                co_rated[x] = [i]
            sum1 += userItemMatrix[u1][i]
            sum2 += userItemMatrix[u2][i]

    if x in co_rated:
        a1, a2 = float(sum1)/len(co_rated[x]), float(sum2)/len(co_rated[x])
        s, sum1, sum2 = 0, 0, 0
        for i in co_rated[x]:
            s += (userItemMatrix[u1][i] - a1)*(userItemMatrix[u2][i] - a2)
            sum1 += (userItemMatrix[u1][i] - a1)*(userItemMatrix[u1][i] - a1)
            sum2 += (userItemMatrix[u2][i] - a2)*(userItemMatrix[u2][i] - a2)

        sum1, sum2 = math.sqrt(sum1), math.sqrt(sum2)
        if s == 0:
            w = 0
        else:
            w = s/(sum1*sum2)
        weight[x] = w
    else:
        weight[x] = "NOCOMMONMOVIES"

def predict(x):
    userID = int(x[0][0])
    movieID = int(x[0][1])
    
    if movieID in usersByItem:
        similarUsers = usersByItem[movieID]
        c, sum = 0, 0
        for key in userItemMatrix[userID].iterkeys():
            sum += userItemMatrix[userID][key]
            c += 1
        r = float(sum)/c

        sum_num, sum_den = 0, 0
        for user in similarUsers:
            if weight[tuple(sorted((userID, user)))] != "NOCOMMONMOVIES":
                w = weight[tuple(sorted((userID, user)))]
                temp, c = 0, 0
                for movie in userItemMatrix[user].iterkeys():
                    if movie != movieID:
                        temp += userItemMatrix[user][movie]
                    c += 1
                temp = float(temp)/c
                sum_num += (userItemMatrix[user][movieID] - temp)*w
                sum_den += abs(w)
                
        if sum_num == 0:
            return ((userID, movieID), r)
        else:
            return ((userID, movieID), (float(sum_num)/sum_den) + r)
        
    else:
        
        return ((userID, movieID), "NIL")

# OBTAIN PREDICTIONS.       
result = sorted(test.map(lambda x: predict(x)).collect())
predictions = []
for i in range(len(result)):
    if result[i][1] == "NIL":
        predictions.append(((result[i][0][0],result[i][0][1]), avgPerUserID[result[i][0][0]]))
    else:
        if result[i][1] > 5:
            predictions.append(((result[i][0][0],result[i][0][1]), 5.0000))
        elif result[i][1] < 0:
            predictions.append(((result[i][0][0],result[i][0][1]), 0.0000))
        else:
            predictions.append(result[i])

# PRINT PREDICTIONS TO OUTPUT FILE.
predictions = sc.parallelize(predictions).sortByKey(True).collect()
myfile = open('Siddhant_Patil_result_task2.txt', 'w')
myfile.write("UserId,MovieId,Pred_rating")
myfile.write("\n")
for p in predictions:
    myfile.write(str(p[0][0])+","+str(p[0][1])+","+str(p[1]))
    myfile.write("\n")
myfile.close()


# OBTAIN ERROR.        
predictions = sc.parallelize(predictions).join(sc.parallelize(train2)).map(lambda x: ((x[0][0],x[0][1]), abs(x[1][0]-x[1][1]))).collect()
c1, c2, c3, c4, c5, s = 0, 0, 0, 0, 0, 0
for t in predictions:
    if t[1]>=0 and t[1]<1:
        c1 += 1
    elif t[1]>=1 and t[1]<2:
        c2 += 1
    elif t[1]>=2 and t[1]<3:
        c3 += 1
    elif t[1]>=3 and t[1]<4:
        c4 += 1
    elif t[1]>=4:
        c5 += 1
    s += t[1]*t[1]

# PRINT VALUES ON TERMINAL.
print ">=0 and <1: "+str(c1)
print ">=1 and <2: "+str(c2)
print ">=2 and <3: "+str(c3)
print ">=3 and <4: "+str(c4)
print ">=4: "+str(c5)
print "RMSE = "+str(math.sqrt(float(s)/len(predictions)))
print "The total execution time taken is " +str(time.time() - start_time)+ " sec."



