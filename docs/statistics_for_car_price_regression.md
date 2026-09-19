# Statistics for Used Car Price Regression

This note focuses on the statistical ideas that are directly useful in the Used Car Price Prediction project.

---

## 1. Mean and Median

The **mean** is the arithmetic average.

The **median** is the middle value after sorting.

For skewed variables, the median can describe the center more robustly because it is less influenced by extreme observations.

In price data, this distinction matters because a small number of expensive vehicles can pull the mean upward.

---

## 2. Variance and Standard Deviation

Variance measures average squared distance from the mean.

Standard deviation is the square root of variance and is expressed in the original units of the variable.

Large spread in variables such as `Present_Price` or `Kms_Driven` can make the dataset visually and statistically heterogeneous.

---

## 3. Covariance

Covariance describes whether two variables tend to move together.

Positive covariance:

```text
X increases → Y tends to increase
```

Negative covariance:

```text
X increases → Y tends to decrease
```

The magnitude is difficult to compare across variable pairs because it depends on their units.

That is one reason correlation is often easier to interpret.

---

## 4. Pearson Correlation

Pearson correlation measures the strength and direction of a **linear** relationship.

Its range is:

```text
-1 ≤ r ≤ 1
```

Interpretation:

- close to `+1`: strong positive linear relationship,
- close to `-1`: strong negative linear relationship,
- close to `0`: weak linear relationship.

In this project, Pearson correlation is used for the numerical features because the baseline model is Linear Regression.

Important:

> Correlation does not imply causation.

---

## 5. Why Raw Categorical Variables Are Not in the Correlation Matrix

Variables such as:

- `Fuel_Type`
- `Seller_Type`
- `Transmission`

are nominal text categories.

A raw Pearson correlation is not meaningful for labels such as "Petrol" or "Diesel".

The correlation matrix therefore focuses on the numeric variables, while the categorical variables are explored with grouped plots and later encoded for modeling.

---

## 6. Outliers and the IQR Rule

The Interquartile Range is:

```text
IQR = Q3 - Q1
```

A common diagnostic flags observations outside:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

In this project the IQR rule is used to **flag** unusual observations.

It is not used as an automatic deletion or clipping rule.

An extreme value can be:

- a data error,
- a rare but valid vehicle,
- or an influential observation.

Those possibilities should be investigated before modifying the data.

---

## 7. Residuals

For observation `i`:

```text
Residual = Actual Value - Predicted Value
```

Residuals show the part of the target that the model did not explain.

A good residual plot should generally show points scattered around zero without a strong systematic pattern.

---

## 8. Homoscedasticity

Homoscedasticity means the residual variance is reasonably constant across the fitted-value range.

If residual spread becomes wider or narrower as predictions increase, that is **heteroscedasticity**.

This can affect classical inference and may also indicate that the linear model is missing structure.

---

## 9. Residual Normality

A common misconception is that the target itself must be normally distributed for Linear Regression.

That is not the key assumption.

For classical regression inference, the normality assumption concerns the **errors/residuals**, especially when using p-values and confidence intervals.

The project uses:

- a residual histogram,
- and a Q-Q plot

to inspect this behavior.

---

## 10. Q-Q Plot

A Q-Q plot compares observed residual quantiles with theoretical normal quantiles.

If points follow the reference line closely, the residual distribution is reasonably similar to a normal distribution.

Large departures, especially in the tails, suggest non-normality.

A Q-Q plot is a diagnostic, not a pass/fail test.

---

## 11. MAE

Mean Absolute Error is:

```text
MAE = average(|actual - predicted|)
```

Advantages:

- easy to interpret,
- same unit as the target,
- less sensitive to large errors than MSE.

Lower is better.

---

## 12. MSE

Mean Squared Error is:

```text
MSE = average((actual - predicted)²)
```

Because errors are squared, large mistakes receive more weight.

Lower is better.

---

## 13. RMSE

Root Mean Squared Error is:

```text
RMSE = sqrt(MSE)
```

It keeps the target's original unit while still emphasizing large errors.

Lower is better.

---

## 14. R²

R² measures how much variation in the target is explained by the model relative to predicting the mean.

A higher value usually indicates better fit, but R² should not be used alone.

A model can have a respectable R² and still have:

- problematic residual patterns,
- unstable coefficients,
- or weak generalization.

---

## 15. Adjusted R²

Adjusted R² modifies R² by accounting for the number of predictors.

It is useful when comparing linear models with different feature counts because adding unnecessary variables does not automatically improve it.

---

## 16. Train/Test Variation

A train/test split is only one sample of possible partitions.

On a small dataset, model performance can change noticeably depending on which observations enter the test set.

That is why this project reports both:

- held-out test performance,
- and 5-fold cross-validation.

---

## 17. Cross-Validation

In 5-fold cross-validation:

1. the data is divided into 5 subsets,
2. the model trains on 4 folds,
3. it validates on the remaining fold,
4. the process repeats until every fold has been used for validation.

The average gives a broader view of performance than one split.

The standard deviation shows how much the score varies across folds.

---

## 18. Why the Whole Pipeline Must Be Cross-Validated

Cross-validating only the final estimator after preprocessing the full dataset can cause leakage.

Instead, the full pipeline is evaluated:

```text
Preprocessing + Encoding + Model
```

This ensures that preprocessing is re-fitted inside each training fold.

---

## 19. Statistical vs. Predictive Interpretation

A coefficient can be statistically interesting without producing a large improvement in prediction.

Likewise, a model can predict reasonably well without satisfying every assumption required for classical statistical inference.

It is useful to separate two questions:

1. **How well does the model predict?**
2. **How reliable are the coefficient-level statistical interpretations?**

This project emphasizes prediction and diagnostic interpretation rather than formal inferential claims.

---

## 20. Project-Specific Takeaways

For this dataset:

- `Present_Price` provides a strong linear signal.
- Vehicle age contributes interpretable depreciation information.
- categorical features require encoding that respects their nominal meaning.
- outlier rules should not be used blindly.
- the held-out split and cross-validation results differ enough that both should be reported.
- residual diagnostics remain important even when R² is reasonably strong.

---

## 21. Interview Questions

**Does the target need to be normally distributed?**  
No. Classical normality assumptions apply to the errors/residuals, not directly to the target.

**Why report both MAE and RMSE?**  
MAE gives an easy-to-understand average error, while RMSE penalizes larger mistakes more heavily.

**Why can cross-validation be more informative than one train/test split?**  
It evaluates the model across several partitions and reduces dependence on one particular split.

**Why not automatically remove every IQR outlier?**  
Because unusual observations may be valid. The IQR rule is a diagnostic, not proof that a row is wrong.

**What does a residual pattern suggest?**  
It may indicate nonlinearity, changing variance, omitted variables, or other structure not captured by the model.
