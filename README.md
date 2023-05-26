# Fast-and-Accurate-pricing-of-volatility-swaps

### Student name: Jorley Johnson
  
Give some brief background and describe what you intend your code to do, together with a rough outline (e.g., classes, functions, snippets, comments, pseudocode).

I will be dividing this project into two phases.

Phase 1:
* Price a volatility swap for spot trading using a list of stocks tracking an index and the stocks' implied volatilities.
  The equation to price the swap is given by: "
```math
  \int_{K = 0}^{S^*} \frac{1}{K^2}e^{rT}p(K)\,dK + \int_{K = {S^*}}^{\infty} \frac{1}{K^2}e^{rT}c(K)\,dK = \sum_{i = 1}^n \frac{\Delta{K_i}}{{K_i}^2}e^{rT}Q({K_i})
```
  Where,
  ```math
  \Delta {K_i} = 0.5(K_{i+1}-K_{i-1}) \forall 2\leq i \leq n-1
  ```
  ```math
  \Delta {K_1} = K_2 - K_1
  ```
  ```math
  \Delta {K_n} = {K_n} - {K_{n-1}}
  ```
  ```math
  Q{K_i}\text{ is the european put option price with strike price } K_i \text{ if } {K_i} < {S^*} \text{ else, it is the eurpoean call option price if } {K_i} > {S^*}. 
  ```
  ```math
  \text{ if } K_i = {S^*}, Q({K_i}) = \text{ average of european call and european put option with strike price } K_i.
  ```
  ```math
  \text{European call option is given by :} \\
  c(K) = S_oN(d_1) - Ke^{-rT}N(d_2)
  ```
  ```math
  \text{European put option is given by : } \\
  p(K) = Ke^{-rT}N(-d_2) - S_oN(-d_1)
  ```
  ```math
  d_1 = \frac{\ln \frac{S_o}{K} + (r + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}
  ```
  ```math
  d_2 = d_1 - \sigma\sqrt{T}
  ```
  * r : risk free interest rate,
  * K : strike price,
  * T : time,
  * S : Stock price,
  * N(x) : is the cumulative distribution function of a variable with a standard normal distribution.
  * ${S^*}$ : is the median value of the list of stock prices
    " [John C Hull, Options Futures and other derivatives, 9th edition, Pearson Education.]

I then go on to create a dataframe of 9 different volatilities populated using the random module as the features and the calculate the Swap price as the target.
Then, I fit the ANN first with a normal fully connected Dense layers using the Keras library.
To compare the performance of this against the twin network, I use the Root Mean Squared Error(RMSE) for determining the error of the model and Explained Variance Score to determine how are the errors dispersed.

Libraries intended to use: Numpy for numerical calculation on arrays, Pandas for storing data in a dataframe and perform statistical analysis on and Matplotlib.pyplot to plot the path of the stocks using Monte Carlo simulation.
