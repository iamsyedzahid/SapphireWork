# Task 2.2: End-to-End Model Pipeline

## Written Interpretation: What Moved the Needle?
Throughout this end-to-end pipeline, the goal was to predict California house values while iteratively improving on a strict baseline. Here is a breakdown of what impacted the model's performance and why:

**1. The Biggest Jump: Moving to a Non-Linear Model**
Our absolute lowest bar was a "dumb baseline" (predicting the mean), which resulted in a massive Test RMSE of `0.9355`. Simply applying Linear Regression dropped the error to `0.5626`. However, the single biggest jump in performance came from switching to a **Random Forest Regressor**, which dropped our baseline 5-Fold CV RMSE to `0.4502`. 
* *Why it moved the needle:* Housing data is highly non-linear (e.g., house prices don't scale perfectly straight with latitude/longitude). A tree-based model was able to capture these complex, non-linear relationships that Linear Regression completely missed.

**2. A Step Backward: Feature Engineering**
I attempted to engineer two new features: `BedroomsPerRoom` and `RoomsPerOccupant`. Surprisingly, this actually *harmed* the model, pushing the CV RMSE up to `0.4508`.
* *Why it moved the needle backward:* The Random Forest was likely confused by the redundancy of the ratio features. It introduced noise that caused the tree to make suboptimal splits. Recognizing this, I dropped the features and proceeded with the raw, cleaned data.

**3. The Final Squeeze: Hyperparameter Tuning**
To get the absolute best performance out of the model, I used `RandomizedSearchCV` on the cleaned dataset. By letting the algorithm test different combinations of `n_estimators`, `max_depth`, and `min_samples_split`, I was able to optimize the architecture of the trees, pushing the CV RMSE down to our final best score of `0.4500`.

**Conclusion:**
By cleaning extreme anomalies early on, validating decisions with 5-Fold Cross Validation instead of single-splits, and tuning the model parameters mathematically, I successfully drove the final Test RMSE down to an impressive **`0.4211`**—a massive improvement over the `0.9355` baseline we started with.