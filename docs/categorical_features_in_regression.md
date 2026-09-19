# Categorical Features in Regression

Categorical variables are common in real-world tabular data, but they need special treatment before they can be used in Linear Regression.

This note explains the decisions used in the Used Car Price Prediction project.

---

## 1. Numerical vs. Categorical Features

A numerical feature has meaningful arithmetic structure.

Examples:

- `Kms_Driven`
- `Present_Price`
- `Vehicle_Age`

A categorical feature represents membership in a group.

Examples:

- `Fuel_Type`
- `Seller_Type`
- `Transmission`

The distinction matters because Linear Regression expects numerical model inputs, but category labels are not automatically numbers in a meaningful mathematical sense.

---

## 2. Why Manual Integer Encoding Can Be Misleading

Suppose fuel type is encoded like this:

```text
Petrol = 1
Diesel = 2
CNG = 3
```

This creates mathematical relationships that do not actually exist.

The model can now treat:

```text
CNG - Diesel = Diesel - Petrol
```

and it may interpret CNG as "larger" than Diesel.

That is appropriate for an ordinal feature only when the order is real. Fuel type is nominal, so that ordering would be artificial.

---

## 3. One-Hot Encoding

One-Hot Encoding creates a separate binary indicator for each category.

For example:

```text
Fuel_Type = Petrol / Diesel / CNG
```

can become:

```text
Fuel_Type_Diesel
Fuel_Type_CNG
```

if `Petrol` is used as the reference category.

A row with Petrol has:

```text
Diesel = 0
CNG = 0
```

A row with Diesel has:

```text
Diesel = 1
CNG = 0
```

A row with CNG has:

```text
Diesel = 0
CNG = 1
```

This lets the model estimate category-specific differences without inventing an ordering.

---

## 4. Reference Categories

When one category is omitted, it becomes the reference category.

In this project the reference categories are:

- `Petrol`
- `Dealer`
- `Manual`

This makes a coefficient such as:

```text
Transmission_Automatic
```

interpretable as the expected difference between Automatic and Manual vehicles, holding the other model inputs constant.

---

## 5. Why `drop="first"` Is Often Used

For a linear model with an intercept, keeping every dummy column creates perfect linear dependence.

For a binary category:

```text
Manual + Automatic = 1
```

If the intercept is also present, one of those columns is redundant.

Dropping one category avoids this exact redundancy and makes the coefficients easier to interpret.

---

## 6. One-Hot Encoding Inside a Pipeline

Encoding should be part of the preprocessing pipeline, not performed manually on the full dataset before splitting.

Why?

Because a pipeline ensures that:

- training and prediction use the same transformation,
- cross-validation fits preprocessing separately inside each fold,
- unseen categories can be handled consistently,
- and preprocessing logic stays attached to the model.

In Scikit-learn, this is commonly handled with:

```python
ColumnTransformer
OneHotEncoder
Pipeline
```

---

## 7. Handling Unknown Categories

A production prediction may contain a category that was not present during training.

Using:

```python
OneHotEncoder(handle_unknown="ignore")
```

prevents the pipeline from failing in that situation.

The unknown category receives zeros across the known one-hot columns for that feature.

This does not mean the model truly understands the unseen category; it simply allows the pipeline to make a prediction without crashing.

---

## 8. High-Cardinality Features

`Car_Name` has many unique values relative to the size of this dataset.

One-hot encoding all of them would create many sparse columns from only a few hundred rows.

Potential consequences include:

- unstable coefficients,
- overfitting,
- sparse design matrices,
- and weak estimates for rare categories.

For the baseline, `Car_Name` is excluded.

Other options for a larger project could include:

- grouping rare categories,
- extracting brand names,
- target-safe encoding,
- embeddings,
- or collecting more data.

---

## 9. Ordinal Categories Are Different

Not every categorical variable is nominal.

For example:

```text
Small < Medium < Large
```

has a real order.

In that case, an ordinal encoding may make sense if the assumed spacing is defensible.

The important rule is:

> The encoding should reflect the meaning of the variable, not just convert text into numbers.

---

## 10. Interview Questions

**Why not encode Fuel_Type as 1, 2, and 3?**  
Because Fuel_Type is nominal. Integer values would create an artificial order and spacing that the categories do not have.

**Why use One-Hot Encoding for Linear Regression?**  
It allows the model to estimate separate category effects without imposing numeric ordering.

**What is the reference category?**  
It is the category omitted from the dummy variables. Other category coefficients are interpreted relative to it.

**Why can high-cardinality one-hot encoding be risky?**  
It creates many sparse features, which can increase variance and overfitting, especially with a small dataset.

**Why place the encoder inside a pipeline?**  
To keep preprocessing consistent across training, cross-validation, testing, and future prediction.
