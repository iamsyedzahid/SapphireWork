# Essential Math Fundamentals for Machine Learning

## Vectors & Matrices

**Explanation:**  
A vector is a list of numbers that represent a point in space or have a direction and magnitude. A matrix is a 2D grid of numbers that represents a linear transformation. Multiplying a matrix by a vector applies that transformation and multiplying two matrices combines their transformations into a single step.

**Worked Example:**  
Multiplying two 2×2 matrices by taking the dot product of the rows of the first matrix with the columns of the second:
```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} (1\times5 + 2\times7) & (1\times6 + 2\times8) \\ (3\times5 + 4\times7) & (3\times6 + 4\times8) \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

## Dot Product

**Explanation:**  
The dot product is an operation that takes two equal length sequences of numbers and returns a single number. Geometrically, it measures how closely two vectors align with each other. A positive dot product means they generally point in the same direction, zero means they are perpendicular, and negative means they point in opposite directions.

**Worked Example:**  
Computing the dot product of `[1, 2, 3]` and `[4, 5, 6]`:
```math
[1, 2, 3] \cdot [4, 5, 6] = (1 \times 4) + (2 \times 5) + (3 \times 6) = 4 + 10 + 18 = 32
```

## Derivative / Gradient

**Explanation:**
A derivative measures the slope or rate of change of a function at a specific point, telling you how much the output changes for a tiny change in input. A gradient is simply the generalization of a derivative to multiple dimensions. It is a vector containing the partial derivatives and always points in the direction of the steepest increase of the function. In machine learning, we use the negative gradient to figure out how to tweak model parameters to reduce errors (gradient descent).

## Mean, Variance, Standard Deviation

**Explanation:**
The **mean** is the average value of a dataset. The **variance** measures how spread out the numbers are from the mean by averaging the squared differences from the mean. The **standard deviation** is the square root of the variance, giving a measure of spread in the same units as the original data.

**Worked Example:**  
For the dataset `[2, 4, 4, 4, 5, 5, 7, 9]`:
*   **Mean**: $(2 + 4 + 4 + 4 + 5 + 5 + 7 + 9) / 8 = 40 / 8 = 5$
*   **Variance**: Calculate squared differences from the mean: $(2-5)^2, (4-5)^2, \dots \rightarrow 9, 1, 1, 1, 0, 0, 4, 16$. 
    Average these: $(9 + 1 + 1 + 1 + 0 + 0 + 4 + 16) / 8 = 32 / 8 = 4$
*   **Standard Deviation**: $\sqrt{\text{Variance}} = \sqrt{4} = 2$

## Probability & Bayes

**Explanation:**  
Conditional probability is the likelihood of an event occurring given that another event has already occurred. Bayes' Theorem gives us a way to update our beliefs based on new evidence. It essentially tells us how to reverse conditional probabilities (e.g., finding the probability of a cause given an effect).

**Worked Example:**  
A disease affects 1% of people. A test for it is 99% accurate (it correctly identifies 99% of sick people, and correctly clears 99% of healthy people). You test positive. What’s the real probability you’re sick?

*   Probability of disease, $P(D) = 0.01$
*   Probability of healthy, $P(H) = 0.99$
*   Probability of testing positive if sick, $P(+|D) = 0.99$
*   Probability of testing positive if healthy (false positive), $P(+|H) = 0.01$

Using Bayes' theorem to find $P(D|+)$:
```math
P(D|+) = \frac{P(+|D) \times P(D)}{P(+|D) \times P(D) + P(+|H) \times P(H)}
```
```math
P(D|+) = \frac{0.99 \times 0.01}{(0.99 \times 0.01) + (0.01 \times 0.99)} = \frac{0.0099}{0.0099 + 0.0099} = \frac{0.0099}{0.0198} = 0.5
```
Despite a 99% accurate test, you only have a 50% chance of actually having the disease! This happens because the disease is so rare that false positives equal true positives.
