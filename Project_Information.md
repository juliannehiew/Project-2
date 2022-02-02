# Lantern Fish birth prediction

## Project Goal

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
 
## Challenges 

Overall a challenging question to answer, with the additional task to incorporate machine learning into the problem; this was not intended for machine learning
Scaling the data, as mentioned before the huge spread of data meant that any scaling would decrease the starting days to practically 0. This meant that only limited windows could be taken before the network would simply ignore the starting values. 
Equipment Challenges
Lack of stronger computer specs, in particular RAM. using the algorithmic solution instead of the value tracking solution would simply take an exponential amount of time and ram. Approaching 200 days, each day would take approx 65 seconds to calculate with my machine’s RAM filling up and rendering he whole notebook inoperable. 
Additional ML:
Using an untaught model was difficult, as was finding a model that would deliberately be ineffective. The Naive Bayes model was perfect, but presented some issues when the values 
were simply too large to fit in. (80 sig figs)

## Conclusion 

The initial question asked to simulate the lantern fish birth rate and calculate the number of lanternfish after 80 days. 
Using machine learning, in particular Tensor Flow and using various training models which were Linear Regression, Naïve Bayes Gaussian Model and LSTM neural network.
Data wrangling is an area of improvement 
LSTM was not efficient with this question, however a simpler network of two neurons would have been enough
Python overflows values in some ML models such as the naïve bayes model. Within the model, it overflowed the integer values at a few hundred trillion fish and stored the data entries as objects rather than integers. 

