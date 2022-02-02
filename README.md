# Puzzle description
## Lanternfish growth prediction
With the algorithmic problem from AoC 2021 day 6;

**--- Day 6: Lanternfish ---**

The sea floor is getting steeper. Maybe the sleigh keys got carried this way?

A massive school of glowing lanternfish swims past. They must spawn quickly to reach such large numbers - maybe exponentially quickly? You should model their growth rate to be sure.

Although you know nothing about this specific species of lanternfish, you make some guesses about their attributes. Surely, each lanternfish creates a new lanternfish once every 7 days.

However, this process isn't necessarily synchronized between every lanternfish - one lanternfish might have 2 days left until it creates another lanternfish, while another might have 4. So, you can model each fish as a single number that represents the number of days until it creates a new lanternfish.

Furthermore, you reason, a new lanternfish would surely need slightly longer before it's capable of producing more lanternfish: two more days for its first cycle.

So, suppose you have a lanternfish with an internal timer value of 3:

After one day, its internal timer would become 2.
After another day, its internal timer would become 1.
After another day, its internal timer would become 0.
After another day, its internal timer would reset to 6, and it would create a new lanternfish with an internal timer of 8.
After another day, the first lanternfish would have an internal timer of 5, and the second lanternfish would have an internal timer of 7.
A lanternfish that creates a new fish resets its timer to 6, not 7 (because 0 is included as a valid timer value). The new lanternfish starts with an internal timer of 8 and does not start counting down until the next day.

Realizing what you're trying to do, the submarine automatically produces a list of the ages of several hundred nearby lanternfish (your puzzle input). For example, suppose you were given the following list:

3,4,3,1,2
This list means that the first fish has an internal timer of 3, the second fish has an internal timer of 4, and so on until the fifth fish, which has an internal timer of 2. Simulating these fish over several days would proceed as follows:

Initial state: 3,4,3,1,2
After  1 day:  2,3,2,0,1
After  2 days: 1,2,1,6,0,8
After  3 days: 0,1,0,5,6,7,8
After  4 days: 6,0,6,4,5,6,7,8,8
After  5 days: 5,6,5,3,4,5,6,7,7,8
After  6 days: 4,5,4,2,3,4,5,6,6,7
After  7 days: 3,4,3,1,2,3,4,5,5,6
After  8 days: 2,3,2,0,1,2,3,4,4,5
After  9 days: 1,2,1,6,0,1,2,3,3,4,8
After 10 days: 0,1,0,5,6,0,1,2,2,3,7,8
After 11 days: 6,0,6,4,5,6,0,1,1,2,6,7,8,8,8
After 12 days: 5,6,5,3,4,5,6,0,0,1,5,6,7,7,7,8,8
After 13 days: 4,5,4,2,3,4,5,6,6,0,4,5,6,6,6,7,7,8,8
After 14 days: 3,4,3,1,2,3,4,5,5,6,3,4,5,5,5,6,6,7,7,8
After 15 days: 2,3,2,0,1,2,3,4,4,5,2,3,4,4,4,5,5,6,6,7
After 16 days: 1,2,1,6,0,1,2,3,3,4,1,2,3,3,3,4,4,5,5,6,8
After 17 days: 0,1,0,5,6,0,1,2,2,3,0,1,2,2,2,3,3,4,4,5,7,8
After 18 days: 6,0,6,4,5,6,0,1,1,2,6,0,1,1,1,2,2,3,3,4,6,7,8,8,8,8
Each day, a 0 becomes a 6 and adds a new 8 to the end of the list, while each other number decreases by 1 if it was present at the start of the day.

In this example, after 18 days, there are a total of 26 fish. After 80 days, there would be a total of 5934.

**--Part One --**

Find a way to simulate lanternfish. How many lanternfish would there be after 80 days?

Your puzzle answer was 363101.


**--- Part Two ---**

Suppose the lanternfish live forever and have unlimited food and space. Would they take over the entire ocean?

After 256 days in the example above, there would be a total of 26984457539 lanternfish!

How many lanternfish would there be after 256 days?

Your puzzle answer was 1644286074024.


We have the input [1,3,3,4,5,1,1,1,1,1,1,2,1,4,1,1,1,5,2,2,4,3,1,1,2,5,4,2,2,3,1,2,3,2,1,1,4,4,2,4,4,1,2,4,3,3,3,1,1,3,4,5,2,5,1,2,5,1,1,1,3,2,3,3,1,4,1,1,4,1,4,1,1,1,1,5,4,2,1,2,2,5,5,1,1,1,1,2,1,1,1,1,3,2,3,1,4,3,1,1,3,1,1,1,1,3,3,4,5,1,1,5,4,4,4,4,2,5,1,1,2,5,1,3,4,4,1,4,1,5,5,2,4,5,1,1,3,1,3,1,4,1,3,1,2,2,1,5,1,5,1,3,1,3,1,4,1,4,5,1,4,5,1,1,5,2,2,4,5,1,3,2,4,2,1,1,1,2,1,2,1,3,4,4,2,2,4,2,1,4,1,3,1,3,5,3,1,1,2,2,1,5,2,1,1,1,1,1,5,4,3,5,3,3,1,5,5,4,4,2,1,1,1,2,5,3,3,2,1,1,1,5,5,3,1,4,4,2,4,2,1,1,1,5,1,2,4,1,3,4,4,2,1,4,2,1,3,4,3,3,2,3,1,5,3,1,1,5,1,2,2,4,4,1,2,3,1,2,1,1,2,1,1,1,2,3,5,5,1,2,3,1,3,5,4,2,1,3,3,4]

From this we can create random lists of random lengths with random numbers in the range of 0 to 8. We can then use machine learning and Neural networks to compute the length of each random list after 150 days. We will then plot the differences between the algorithm and machine learning/neural networks.


*****************************************************************************************************************************************************************************

# Project Goal

*In order to predict the exponential birth rate of the lantern fish, we aim to use various machine learning models to evaluate the number of fishes within a given time frame* 

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

## Results



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

