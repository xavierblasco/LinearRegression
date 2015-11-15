#import the numpy Library
import numpy as np
#import lstsq -> Return the least-squares solution to a linear matrix equation.
from scipy.linalg import lstsq


#****************************************************************************************
#Q1 Download Housing Data Set from the UCI repository


#****************************************************************************************
#Q2 Load the data (using numpy.loadtxt) and separate the last column (target value, MEDV). Compute the average of the target value and the MSE obtained using it as a constant prediction.


#Load data from 'housing.data' file placed on the same directory
A=np.loadtxt('housing.data')

#Separate last column : gives all values [row,col]
y = A[:,-1]

#print number of points
print ""
print ""
print "Q1 - Download Housing Data Set *************"
print 'Number of data points: ', len(y)
print ""
print ""

#Dummy features -> Creates an array of len(y) values and set 1 for all of them
#We use it as bias vector
X = np.ones((len(y),1))

#Shortcuts for numpy functions
# np.dot -> product of two arrays
# np.linalg.inv -> inverse
dot = np.dot
inv = np.linalg.inv

#Find Theta= ((X'*X)^-1)*X'*y
theta = dot(dot(inv(dot(X.T, X)), X.T), y)
print "Q2 ****************************************"
print "theta = ((X'*X)^-1)*X'*y "
print "theta = ", theta

#MSE = (1/N)*sum((y-X*theta)^2) 
MSE = (sum((y-dot(X, theta))**2))/len(y)
print "MSE = (1/N)*sum((y-X*theta)^2) "
print "MSE = ", MSE
print ""
print ""


#****************************************************************************************
#Q3 Split the data in two parts (50%-50%) for training and testing (first half for training, second half for testing). Train a linear regressor model for each variable individually (plus a bias term) and compute the MSE on the training and the testing set. Which variable is the most informative? which one makes the model generalize better? and worse? Compute the coefficient of determination (R^2) for the test set.

#    Hint: If you want to select the i-th column of an array, but want it to retain the two dimension, you can do it like that:
#
#    column = data_array[:,i:i+1] 

total_points = len(y)
num_rows_1 = np.floor(len(y)/2) 	#Rows for first 50% of records
num_rows_2 = total_points - num_rows_1  #Rows for second 50% od records

print "Q3 ****************************************"
print 'Number of data points first 50%: ', num_rows_1
print 'Number of data points second 50%: ', num_rows_2

A1 = A[:num_rows_1,:-1] 	#Get All data of first 50% of rows
y1 = A[:num_rows_1,-1]		#Separate last column of first 50% of rows

A2 = A[num_rows_1:,:-1] 	#Get All data of second 50% of rows
y2 = A[num_rows_1:,-1]		#Separate last column of second 50% of rows

# Let's add a continuous variable, for each variable (i= col of variable)
num_of_variables = A1.shape[1] 	#Get number of columns for A1 Array.
train_theta =[]	#Create a new array for each theta. One for each variable
train_MSE =[]	#Create a new array for each training MSE. One for each variable
test_MSE =[]	#Create a new array for each testing MSE. One for each variable

for i in range(0, num_of_variables):
	X1 = np.hstack((X[:num_rows_1], A1[:,i].reshape(len(y1),1))) 	        #training add variables with bias values	
	train_theta.append(lstsq(X1,y1)[0]) 					#Get theta for each individual variable in training array
	#print  "train_theta[",i,"]= ",lstsq(X1,y1)[0]
	train_MSE.append((sum((y1-dot(X1, train_theta[i] ))**2))/len(y1))	#Get MSE for each individual variable on training array
	X2 = np.hstack((X[num_rows_1:], A2[:,i].reshape(len(y2),1))) 	        #testing add variables with bias values
	test_MSE.append((sum((y2-dot(X2, train_theta[i] ))**2))/len(y2))	#Get MSE for each individual variable on testing array
	print  "Training MSE[",i,"] = ",train_MSE[i], " Testing MSE[",i,"]",test_MSE[i]
	

print  ""
print  "Most informative variable= ",train_MSE.index(np.amin(train_MSE)), " with MSE[",train_MSE.index(np.amin(train_MSE)),"] = ",np.amin(train_MSE) 		#Shows Most informative variable. Calculate min value of MSE . Minimum error is the best.
print  "Variable that generalize better the model = ",test_MSE.index(np.amin(test_MSE)), " with MSE[",test_MSE.index(np.amin(test_MSE)),"] = ",np.amin(test_MSE) 	#Shows the variable that better generalize the model. Minimum MSE in testing set because minimum error is better.
print  "Variable that generalize worse the model = ",test_MSE.index(np.amax(test_MSE)), " with MSE[",test_MSE.index(np.amax(test_MSE)),"] = ",np.amax(test_MSE) 	#Shows the variable that worse generalize the model. Maximum MSE in testing set because maximum error is worse.

print ""
print "Coefficient of determination (R^2) for the test set********"
#mean = (1/N)*sum(y)
mean = sum(y2)/len(y2)
print "mean as sum(y2)/len(y2) = ", mean
mean2 = y2.mean()
print "mean as y2.mean() = ", mean2

var = np.mean(abs(y2 - mean2)**2)		 #Calculate Var using academic expression
print "var as mean(abs(x - x.mean())**2 = ", var 
var = y2.var()					 #Calculate Var using np expression 
print "y2.var() = ", var

#FVU = MSE / var
FVU = test_MSE/var				#Calculate FVU, we get an array of FVU , on for each estimation previously
print "Array of FVU for each estimation done"
print "FVU as test_MSE / var = ", FVU
R2 = 1- FVU					#Calculate R2, we get an array of R2 , on for each estimation previously

print ""
print "Array of R2 for each estimation done"
print "Coefficient of determination (R^2) for the test set= ", R2
print ""
print ""


#****************************************************************************************
#Q4 Now train a model with all the variables plus a bias term. What is the performance in the test set? Try removing the worst-performing variable you found in step 2, and run again the experiment. What happened?


X3 = np.hstack((X[:num_rows_1], A1[:,:].reshape(len(y1),num_of_variables))) #training add all variables with bias values	
train_theta_3 = 0	#Create a variable for theta called train_theta_3 for train a new model with all variables
train_MSE_3 = 0		#Create a variable for training MSE called train_MSE_3 for train a new model with all variables
test_MSE_3 = 0		#Create a variable for testing MSE called train_MSE_3 for train a new model with all variables
train_theta_3 =(lstsq(X3,y1)[0])  #Get theta and save on train_theta_3 variable
train_MSE_3 = ((sum((y1-dot(X3, train_theta_3 ))**2))/len(y1)) #Get MSE for model with all variable on training set
test_MSE_3 = ((sum((y2-dot(X3, train_theta_3 ))**2))/len(y2)) #Get MSE for model with all variable on testing set

print "Q4 ****************************************"
print  "Training MSE for model with all variables on training set = ",train_MSE_3
print  "Training MSE for model with all variables on testing set = ",test_MSE_3
#Mean and var is the same tah Q3 because is based just on y2
FVU_3 = test_MSE_3/var	
R2_3 = 1-FVU_3
print  "FVU for model with all variables on testing set = ",FVU_3
print  "R2 for model with all variables on testing set = ",R2_3
#R2 Coefficient of determination should be on 0-1 range for model estimated. But when we apply this parameters to test set we can obtain results out of range, like this case happends. Model is so worse that R2 is negative

#Try now removing worse variable of previous estimations

index_worse = test_MSE.index(np.amax(test_MSE)) #save ina variable worse estimation index (number of worse variable) 


#With following expresssion we can get new array of variables for any index of worse variable
if index_worse == 0:
	A3 = A1[:, 1:] 	#Get All data of second 50% of rows least first one (worse) on training set	
	A4 = A2[:, 1:] 	#Get All data of second 50% of rows least first one (worse) on testing set
elif 0 < index_worse and index_worse < num_of_variables:
	A3 = A1[:, range(0,index_worse)+range(index_worse+1,num_of_variables)] 	#Get All data of second 50% of rows least first one (worse) on training set	
	A4 = A5[:, range(0,index_worse)+range(index_worse+1,num_of_variables)] 	#Get All data of second 50% of rows least first one (worse) on testing set
elif index_worse == num_of_variables:
	A3 = A1[:, :-1] 	#Get All data of second 50% of rows least first one (worse) on training set	
	A4 = A5[:, :-1] 	#Get All data of second 50% of rows least first one (worse) on testing set



X4 = np.hstack((X[:num_rows_1], A3[:,:].reshape(len(y1),(num_of_variables-1)))) #training add all variables with bias values	
train_theta_4 = 0	#Create a variable for theta called train_theta_4 for train a new model with all variables  least worse
train_MSE_4 = 0		#Create a variable for training MSE called train_MSE_4 for train a new model with all variables least worse
test_MSE_4 = 0		#Create a variable for testing MSE called train_MSE_4 for train a new model with all variables  least worse
train_theta_4 =(lstsq(X4,y1)[0])  #Get theta and save on train_theta_4 variable
train_MSE_4 = ((sum((y1-dot(X4, train_theta_4 ))**2))/len(y1)) #Get MSE for model with all variable  least worse on training set
test_MSE_4 = ((sum((y2-dot(X4, train_theta_4 ))**2))/len(y2)) #Get MSE for model with all variable  least worse on testing set
print  ""
print  "Training MSE for model with all variables (least worse) on training set = ",train_MSE_4
print  "Training MSE for model with all variables (least worse) on testing set = ",test_MSE_4
#Mean and var is the same tah Q3 because is based just on y2
FVU_4 = test_MSE_4/var	
R2_4 = 1-FVU_4
print  "FVU for model with all variables on testing set = ",FVU_4
print  "R2 for model with all variables on testing set = ",R2_4

print ""
print "Previous R2: ", R2_3, " New R2: ", R2_4
print "New Model is similar to previous model"



			
















	











