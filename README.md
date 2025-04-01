### Forecasting the number of antidiabetic drug prescriptions in Australia

- We will be using timeseries techniques to solve this project problem.

#### Objective:
- The objective of the project is forecasting the number of antidiabetic drug prescriptions in Australia, from 1991 to 2008.
- To solve the problem of overproduction of the antidiabetic drugs by predictioning how much is needed.

#### Steps
- a. Visualize the timeseries
- b. Use timeseries decomposition to extract trend,seasonality and the residuals.
- c. Determine the most suitable model
- d. model the series with:
    - apply transformation to make it stationary
    - set the values of *d* and D. set the value of *m*
    - find the optimal (p,d,q)(P,D,Q)*<sub>m</sub>*
    - perform residual analysis to validate the model.
- e. Perform rolling forecast of 12 months on the test set.
- f. Visualize your forecasts.
- g. Compare the model's performance to a baseline 

### Results 

- After performing decomposition on the data we get the following componets: Trend , seasonality and the residuals. Which help in determing which model to use:
![decomposition](../Antidiabetic/results/stl)