# Puzzle description : Lanternfish growth prediction
**With the algorithmic problem from AoC 2021 day 6**;

https://adventofcode.com/2021/day/6

A massive school of glowing lanternfish swims past. They must spawn quickly to reach such large numbers - maybe exponentially quickly? You should model their growth rate to be sure.   Although you know nothing about this specific species of lanternfish, you make some guesses about their attributes. Surely, each lanternfish creates a new lanternfish once every 7 days.

However, this process isn't necessarily synchronized between every lanternfish - one lanternfish might have 2 days left until it creates another lanternfish, while another might have 4. So, you can model each fish as a single number that represents the number of days until it creates a new lanternfish.


So, suppose you have a lanternfish with an internal timer value of 3:

* After one day, its internal timer would become 2.
* After another day, its internal timer would become 1.
* After another day, its internal timer would become 0.
* After another day, its internal timer would reset to 6, and it would create a new lanternfish with an internal timer of 8.
* After another day, the first lanternfish would have an internal timer of 5, and the second lanternfish would have an internal timer of 7.
A lanternfish that creates a new fish resets its timer to 6, not 7 (because 0 is included as a valid timer value). The new lanternfish starts with an internal timer of 8 and does not start counting down until the next day.

Realizing what you're trying to do, the submarine automatically produces a list of the ages of several hundred nearby lanternfish (your puzzle input). For example, suppose you were given the following list:

3,4,3,1,2
This list means that the first fish has an internal timer of 3, the second fish has an internal timer of 4, and so on until the fifth fish, which has an internal timer of 2. Simulating these fish over several days would proceed as follows:


## ![rate](https://github.com/juliannehiew/Project2-Lanternfish-Growth-Prediction/blob/main/images/Lanternfish%20rate.JPG)

Each day, a 0 becomes a 6 and adds a new 8 to the end of the list, while each other number decreases by 1 if it was present at the start of the day.

In this example, after 18 days, there are a total of 26 fish. After 80 days, there would be a total of 5934.




*****************************************************************************************************************************************************************************

# Project Goal

*In order to predict the exponential birth rate of the lantern fish, we aim to use various machine learning models to evaluate the number of fishes within a given time frame* 

**The python file is called Project Notebook.ipynb**



## Model Used

Tensor Flow
Generally a stronger and more reliable deep learning library with excellent data visualization capabilities.
The nature of our data being exponentially large, having a machine learning model such as tensor flow which is scalable was important in choosing a model. 

## Data Preparation 
Initially, it was difficult to prepare the data for the machine learning model due to the exponential growth of the fish; with our test set for example, by day 2500 the number of fish exceed 80 digits long. 
To cater for this, we limited the number of days to predict the amount of fishes to 150 days. Although scalers was an option, some such as the minmax scaler which scales all values between 0 and 1 would result in the initial values being reduced.

To ensure that python would be able to read the exponentially increasing data being put through our machine learning model, we used a random number generator for x amount of fishes where a value of the amount of fishes at the end of the day would be given. From here, we are able to calculate the end number of fish in a given time period.
This allows python to sequentially run through the data without being “overloaded” with the exponential data that would otherwise have to be calculated through brute force, which had lead to an error.

## Model Taining

Utilized the following modelling training methods and models:
Linear Regression
Naïve Bayes Gaussian Model
LSTM neural network
These libraries allowed us to  calculate the difference of sum of each day. 
The Naïve Bayes Gaussian Model performed the worst; it assumes that each value is independent from the previous data point, but due to the nature of our data, it would naturally not fit. 

## Evaluation of model performance 

Using sklearn.metrics and its mean square error and r2_score evaluation libraries, we assessed the performance of the model. 
Assessing the results (Generated Increase vs Predicted Increase) with an out of sample and in sample RMSE (Root mean squared error) yielded the following figures:
Out of sample RMSE = 1.3989
In sample RMSE =0.93125

## Results with N values

Due to the linerality of the nature of linear regression, the more N values we take, the predicted results started to shift away from the linear pattern and more into a curve.


## ![N=25](https://github.com/juliannehiew/Project2-Lanternfish-Growth-Prediction/blob/main/images/N%3D25.JPG)


## ![N=50](https://github.com/juliannehiew/Project2-Lanternfish-Growth-Prediction/blob/main/images/N%20%3D%2050.JPG)


## ![N=75](https://github.com/juliannehiew/Project2-Lanternfish-Growth-Prediction/blob/main/images/N%20%3D%2075.JPG)


## ![N=100](https://github.com/juliannehiew/Project2-Lanternfish-Growth-Prediction/blob/main/images/N%20%3D%20100.JPG)


## ![N=150](https://github.com/juliannehiew/Project2-Lanternfish-Growth-Prediction/blob/main/images/N%20%3D%20150.JPG)


## ![N=200](https://github.com/juliannehiew/Project2-Lanternfish-Growth-Prediction/blob/main/images/N%20%3D%20200.JPG)


## ![Naive Bayes Model](https://github.com/juliannehiew/Project2-Lanternfish-Growth-Prediction/blob/main/images/Naive%20Bayes.JPG)


## ![LTSM](https://github.com/juliannehiew/Project2-Lanternfish-Growth-Prediction/blob/main/images/LTSM.JPG)



## Challenges 

* Unfortunately, after running for a few minutes, it crashed with several errors.

* What happens is that there are too many fish, and computing the collection of timer values for every one of them eats all the memory of my computer!

* In such a situation, we need to come up with a different model. We need to find a model that can compute the same result, but by using less information.  If we look at the computation, there are lots of redundant parts. For every fish whose timer value is initially the same, the sequence of computations will also be the same to find out how many fish will be created by it and its descendants, within 256 days.  Therefore, we need to find a model to compute it effieciently.
  
* In short, this was not intended for machine learning. As mentioned before the huge spread of data meant that any scaling would decrease the starting days to practically 0. This meant that only limited windows could be taken before the network would simply ignore the starting values. 

* With the lack of stronger computer specs, in particular RAM. using the algorithmic solution instead of the value tracking solution would simply take an exponential amount of time and ram. Approaching 200 days, each day would take approx 65 seconds to calculate with my machine’s RAM filling up and rendering he whole notebook inoperable. 

* Furthemore, applying an untaught model was difficult and challenging. The Naive Bayes model was perfect and simple, but presented some issues when the values were simply too large to fit in. (80 sig figs)


## Conclusion 

 
* Machine learning was applied, in particular Tensor Flow and various training models which were Linear Regression, Naïve Bayes Gaussian Model and LSTM neural network.
Data wrangling is an area of improvement 

* LSTM was not efficient with this question, however a simpler network of two neurons would have been enough.  

* Python overflows values in some ML models such as the naïve bayes model. Within the model, it overflowed the integer values at a few hundred trillion fish and stored the data entries as objects rather than integers. 

