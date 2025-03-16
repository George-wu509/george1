
**

## A/B Testing

### Test on the Differences

- When testing the difference of means between two distributions, the mean difference typically follows a symmetric distribution
    
- For a mean distribution to achieve symmetry, it requires a smaller sample size when the original distribution is symmetric.
    
- The sample size required to achieve symmetry is roughly 355*s**2, which s = skewness
    

### Early Stopping

- Early stopping in A/B testing means ending an experiment early based on interim results, which impacts the probabilities of Type I and Type II errors.
    
- One method for continuous inspection is dividing the experimental period into sessions and applying Bonferroni correction or FDR based on the number of sessions.
    
- There are other methods that are good for continuous inspection
    

### P-value

- Definition: The p-value is the probability of observing a test statistic at least as extreme as the one obtained from your sample data, assuming that the null hypothesis is true 
    
- Plain Language: It indicates the likelihood of obtaining your current result by chance if the variable you're testing has no actual effect
    

### Power

- Definition: Statistical power is the probability that a test will correctly reject a false null hypothesis
    
- Plain Language: Statistical power is the chance that a test will find a real effect if there is one
    

### Power Analysis

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdjCE265Q-5sqhS-jfbnfnwW_Y81-OeM06ABCWrjHo-Ue6Xc2zNicZ1u9IyO53LoEIS_mUIcKn7p507CvpOOIVdNchCfFsRpUaKOh1WJj3zPGsMoIJZc7TIRAhwl9Zcwq6_HLAV7INPH8Gmtp-8Af_NIdkkEu2j4b5SR2GtCamz-FhRbAP-QJE?key=IXpe4oV-OtSIbW3o2M5jrA)

or roughly 16*((SD)/(MDE))**2

### 3 Assumptions of Experimentations

- Exchangeability (or No Unmeasured Confounding):
    

- This assumption states that the treatment and control groups are comparable in terms of both observed and unobserved variables that can influence the outcome. In other words, the only systematic difference between the groups is the treatment itself.
    

- Positivity (or Overlap):
    

- This assumption ensures that every individual in the population has a positive probability of receiving each level of the treatment. It means there are no covariate patterns where the probability of receiving the treatment is zero  
    or one.
    

- Strong Ignorability is the combination of Exchangeability and Positivity
    

- Consistency (or Stable Unit Treatment Value Assumption, SUTVA):
    

- The potential outcomes for any unit do not vary with the treatments assigned to other units
    
- For each unit, there are no different forms or versions of each treatment level
    

### Common Pitfalls

- Unobserved confounding factors
    
- Deceptive or spurious correlation
    

- Stronger outliers
    

### 3 Types of T-tests

- One-Sample t-test
    
- Independent Two-Sample t-test
    
- Paired Sample t-test
    

#### T-test Assumptions

- Independent Two-Sample and paired sample t-test: 
    

- Normality: the data should be approximately normally distributed.
    
- Independence: the data collected from one subject should not influence the data collected from another.
    

- Homogeneity of variances
    
- Continuous data
    
- Random Sampling
    

## Machine Learning

### Imbalanced Classification

#### Precision vs Recall

- Precision: TP/(TP+FP), among the positive labels what is the pct that is true positive
    
- Recall: TP/(TP+FN), what pct of true positive is correctly labeled
    

  

The False Positive Rate (FPR), calculated as FP/(FP+TN), includes true negatives (TN). When TN is large, the FPR becomes very small, leading to an inflated True Positive Rate (TPR) in the conventional AUC. In contrast, the Area Under the Precision-Recall Curve (AUC-PR) is not affected by TN, making it a better metric for highly imbalanced data.

#### Typical Case: ROC Curve 

The Receiver Operating Characteristic (ROC) curve is a graph showing the performance of a classification model at all classification thresholds. The curve plots two parameters:

- True Positive Rate (TPR): Also known as Sensitivity or Recall, it is the ratio of correctly predicted positive observations to all actual positives. TPR=TP/(TP+FN)​.
    
- False Positive Rate (FPR): It is the ratio of incorrectly predicted positive observations to all actual negatives. FPR=FP/(FP+TN)
    

It is good because:

- It is threshold-independent
    
- It considers both the sensitivity and specificity of the model so is a balanced performance measure.
    

#### Highly Imbalanced Classes

Why Precision-Recall AUC or Average Precision are better than typical AUC: with highly imbalanced classes, the number of TN is very large, so the score will be inflated. PR AUC or Average Precision does not include any TNs so are the better solutions.

- AP is a weighted average method that emphasizes the precision performance at high recall levels, allowing it to reflect the model's performance more delicately.
    
- PR AUC is a simple geometric integration that measures the area under the entire Precision-Recall curve, but it does not weight different recall levels, making it potentially less precise compared to AP.
    

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfaY5lpf15VTjtVJs1z_vitJsRT10VhSH7tX4S9hCmMSbrJNrSouuT3VrIfzgOknbZoJ0avL93KO6GbzI731YShn-ukZpLpCyfdN57bEJc6apAgUx0Plx-hMpCrm21_ocKiHS1v_6jM-fz7JTbDNvLas_YHuEvKfA75h_WxF6nT-RIP7JTWUC8?key=IXpe4oV-OtSIbW3o2M5jrA)

### Bias vs Variance

Bias: Bias measures how much the predictions of the model differ from the true values.

Variance: Variance measures the variability of the model's predictions for different training sets. Underfitting (High Bias) vs Overfitting (High Variance)

  

### Model Comparison

#### Bagging vs Boosting

Bagging is better when overfitting (high variance) is a problem, and Boosting is better when underfitting (high bias) is the main issue

  

However, since models can always be tuned, boosting generally outperforms bagging in most scenarios.

#### GBT vs Deep Learning

GBT usually requires minimal tuning to achieve the same predictive accuracy as deep learning models, making it a popular choice for solving practical problems.

## Product

### E-Commerce Metrics Reference

https://opaque-cerise-fc7.notion.site/Pinterest-Research-f67a8033b91c41b2bb3ad595590ecc7a

### Metric Design Notes

- Define the type of impact the business case care, and spend extra time identifying target units
    
- If a metric is continuous and has high variance, convert it to a proportion if possible, or capped it.  Also avoid overly insensitive metrics (e.g. DAU). 
    
- Always consider cannibalization.
    

#### Marketing

Return On Advertising Spend, (ROAS) = Return/Spend

## Statistics

### Parametric vs Non-parametric Tests

- Use Parametric Tests: generally more powerful and can detect smaller differences or effects
    

- Assumption of normality
    
- Homogeneity of variances
    
- Continuous data
    
- Example: t-test. Welch’s t-test
    

- Use Non-Parametric Tests: less sensitive to outliers and skewed distributions 
    

- Non-normality assumptions
    
- Ordinal or nominal scale
    
- Example: 
    

- KS test: if one from a certain distribution, or two were from the same distribution
    
- Wilcoxon–Mann–Whitney (WMW) test: if there is a difference of median when the dependent variable is either ordinal or continuous but not normally distributed
    

- Non-Parametric Alternatives:
    

- Monte Carlo simulation
    

### Central Limit Theorem (CLT)

When the following conditions are met, the distribution of the sample mean will follow a normal distribution:

- The sample size is large.
    
- The sample must be drawn randomly, and independent of each other.
    
- For the same sample size, samples that follow a normal distribution will have a narrower confidence interval than those from other distributions. This is because skewness and kurtosis in non-normal distributions could increase variance.
    

### Bayesian Equation

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdJfs1ZTY5mImwGnELeKzchu0F3HCucGhyHlzNZ7-uPnhGYn66_r3QQvNLMLNQNLpGMEiJf2vnmIOeLfZcK6apwOJcXLh3HDlg4JbqaU8rLaOxKhuRppoBmGcMNzB7ncS-Oj6AxHfdnv0rBLk81ZYkdIiae4rXBDpPVPTK7UxbMqiG9KQTu0lI?key=IXpe4oV-OtSIbW3o2M5jrA)

### Linear Regression

#### Assumptions

- Linearity
    
- Independence of observations (row independence)
    
- Normality of residuals
    
- Even variance of residuals, if violated:
    

- Log or Box-Cox transformation for target
    
- Transforming predictors
    
- WLS, GLS
    

- No (little) multicollinearity (column independence)
    

#### Test Stats

- R2: 
    

- Only meaningful for prediction purpose
    
- Does not affect the significance of the regression terms if used for analyzing experiment result
    

- t-stats: if the term has significant impact or not
    
- F-stats: the significance of difference between an intercept model (a model with only intercept term) and the current model
    

#### Error and Residual

- Error: the difference between the actual and true model prediction.
    

- Error includes the deviation caused by random factors or unpredictable influences that the model cannot capture or explain.
    

- Residual: the difference between the actual observed value and the value predicted by the estimated model.
    

- Residuals are used to assess how well the model fits the data.
    

#### Correlation and R2

Correlation

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfVQtrCSCdUfm0V3LPesZsU0LOChJRvSwbupcZMSe6YqBos3ZvAuVBvp3wWX3ory-HQC5ZEk08xfRPlLIa4KwO0bgbZtg5nDGPW6ULGLhq00p2RRQsUJQ36G4svZv4QMNAn2gUV79VT3aPL6e8Ukc52V-zmBC5eB_0vPnW76d3d0ZJ-9M84Omo?key=IXpe4oV-OtSIbW3o2M5jrA)

R2

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcVdLeRLGfX2sXnOvZNAkx7DUEkj1FpIgJG8W9ELQHf3HKrrsXeHx-BBYnT8fMsSZkYJPA2AoIzKLGdWABVoWo_-pGeQZ-dyeGonauThORABg1BKdlghWSa-2EbFluyrML8rOxx3Nm1gYu5cDfU_Xtd_hxTeUI7irRxu0SQjRJYr8d4dzut26Y?key=IXpe4oV-OtSIbW3o2M5jrA)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfGqaLDLdrGBIhrFwnwOtwLdknnYbqPLlQs3msPXDVJO2ivmyCApBBveq5uCqYsas3lbHW29ZKgxNyxRI0f5-Af5_l7f3ZqHlL7zvEUlHhlNKHZ17k5TwxQOzJPY2a4JFnGNbrkHf1fBjVMVNm2O6kqypdv5fANX3p0s9rteCBx6q1iwg4i1po?key=IXpe4oV-OtSIbW3o2M5jrA)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdbWm889dqMZZ72r1seXeuw1KdZCVCHMjXff5YVdl6HXsraX1Ty8LN6kxmQ1_e82rw-CmcE0YuSKeibhkTILy2K3FzVQLX9gVI3E4WXz6oodavaJxHaDX1-4oGzBIvGkBKjanmyNXHz1DyaSXPoSPDK2EZ7qgie0cP9NvDnFQ79gCOwqAUBo2E?key=IXpe4oV-OtSIbW3o2M5jrA)

  

#### Coefficient

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXenltfskhIPdQY4pP3UM-yt7go8axDO9IyhRnn4e9sJuHc1ib7a0ex4LDiBu-7ybVpnTtEF7NxroH4Uv_F1GN4Z8rwkTvy-rhxyNDrkkGs0b0YR20N1-VtzPzK_KZj8FsHiu4WUXKDJTOl6XjD94YAgqvokHTtt0heqV4FA2SQCXcINuyXVTWI?key=IXpe4oV-OtSIbW3o2M5jrA)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXezdECCZulYMJZ2RLjmPyAYciOb5b3KRWuj9Q5m6MUjIl1CuCzS1Fp8EBrZX2vPiKRzVJiniAF2rYHyCQoHmeym_BdjL0DIJLHLgx6s-YBgjkMlMibyQ0Gk4ry3uRY7XPK04Lq8RPCIHYQzzfWlo7orGNYjJ5Q7RRu9kaJQerVsSRSAqaV3gw?key=IXpe4oV-OtSIbW3o2M5jrA)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfIxwjvxJJHrohXmSZY1UkeM7D0WGSjNTY3mWNE4VpfjeOfaWByf5NLV2-j6thHEKkSKnnku8T7LNXfA4lLbUvZD8Qgo48ZYVMy_A1i5efPvTESFiDEyD881H9pL7iN-9pXA8FH7Mu--otPBsRyTDe6LikUjtOpqE6H09q066R642xyDHrotvk?key=IXpe4oV-OtSIbW3o2M5jrA)

#### Lasso and Ridge

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfXWY_cQaXHR8_9TrYljiC-5HfRJkU_YoEmkollP1afAk98pFtq8gRvC-A8mvuZf0o8tKmWiXNquqkBt-wx1XVjg9xyhhXquzOkYU2FTgvgLpGEHxCQOq2XWWQdtsfnLzZIyMyJ8cnK2g5KdZ9koSwdRIE17Utm5xWeCm_YJZ25mFOYpIbL2Q?key=IXpe4oV-OtSIbW3o2M5jrA)

#### Colinearity and VIF

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXefx8B3DzfNHKc1xsCDiAEVnvNaoTAYZFmCAuxPBVCGsrHslvuG7NWYbFV2gp1pWsJfF7Q17CA_-aMd2vq6WwvwnmkSfpnK15M-FOxPJ8BkIG3i4LOabu_tvLL1xWoDcqAm4TaljWk9pBU1P5RbFzOt_EHq-TZ4SzXD2Os_OWu7B---tN-C85Q?key=IXpe4oV-OtSIbW3o2M5jrA)

- Instability of Coefficients
    

- Large standard errors
    
- Non-intuitive signs (directional contradict expectation or common sense)
    
- Inflated magnitudes
    

- Difficulty in Interpreting Coefficients
    
- Overall model predictions (fitted values) may still remain relatively accurate, but sensitive to changes in input data, so will reduce generalizability
    
- The coefficient estimations of features without strong collinearity will still be impacted by the correlated features:
    

- Insatiable coefficient estimations
    
- Interpretation difficulties
    
- Reduction of predictive ability 
    

#### Missing Inputs

- Mean/Median Imputation: 
    

- Simple and quick but may distort data if missing values are not random.
    
- Good for Linear-based models since it does not impact data point leverage too much
    

- KNN Imputation: more robust, especially for datasets with complex relationships
    
- Interpolation or Back/Forward fills: good for time-series data 
    
- New category: For categorical columns, create a separate category for missing values.
    
- Unreasonable number: for tree models only, impute missing values in continuous columns with an unreasonable number.
    

### Binomial and Bernoulli Distribution

#### Bernoulli

Mean: p = 1*p+0*(1-p)

Var: E(Y**2) - (E(Y))**2, since Y in [0,1] so Y**2 = Y, so E(Y**2) -(E(Y))**2 = p - p**2 = p(1-p) 

#### Binomial

Mean: np

Var: np(1-p)

- Var(X) = Var(Y1) + Var(Y2) + … + Var(Yn), binomial variable X is composed of Y1, Y2.. Yn bernoulli distributions 
    
- Var(Yi) = p(1-p), so Var(Y1) + Var(Y2) + … + Var(Yn) = np(1-p)
    

  

### Normality Approximation

X follows Binomial Distribution, and X/n = p is the proportion metric we need, so based on Binomial Distribution:

Mean: E(X/n) = np/n = p

Var: Var(X/n) = 1/(n**2)*Var(X) = 1/(n**2)*np*(1-p) = p*(1-p)/n

  

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdJCFSbKdSY72L_CvsnhbNIK-nYgCXuYDr78Ed_5z-W7-eMSqDfKHkjDtTk5QjnhtiugfDUxhpBj7sB4sMAITsiAjQfxiib3O9H9pAncnfbKFTON75Qw2LzLD9jdjCT2AfinbWr2_tKsfE1LtM8b4_qZi8ObMOhmhkmfM0ApwfiRy5CCUPj3Yw?key=IXpe4oV-OtSIbW3o2M5jrA)

This is a Z-test, and there is no proof that a t-test is better.

## Causal Inference

### Heterogeneous Treatment Effect

A typical workflow of HTE is:

- Conduct an A/B test and then use the experiment data to train the HTE model
    
- Identify the optimal treatment for different segments 
    
- Run a second experiment with the personalized treatment to validate the ideas
    

### Causal Inference Concept Chart (from Uber)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdeJDcLYeZuCFDwHLFx_GC5UzQxvqyBboiFPgSuZ2RQ0Pl3ix95i1v-FHhOstd-gNa2_XdJ6cvmYk-6LlpOrYyDywmNaXsHJqeV5-QFTuQ5h_WMgKJJ3AWSUdWPnPpW2VlTlrPt1oGGW0wwmGI6w3omFeOr5uAH3HbUYWpEvVjp_6T34wjoL7g?key=IXpe4oV-OtSIbW3o2M5jrA)

### Assumptions of Common Causal Inference Methods

Linear Regression:

- The treated and non-treated are comparable
    
- All confounding factors are observed
    

Propensity Score Series:  make the treatment and non-treated comparable

- All confounding factors are observed
    

Regression Discontinuity Design: the treatment applied based on a threshold generated by some mechanisms

- No confounding factor sharing the same threshold 
    

Instruments: hard to find in the world of business applications

- When some confounding factors are not observed
    

Diff in Diff: unobserved confounding factors exist and no good quality instruments

- The treated and non-treated follow the similar trends
    

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcHHXYc9vYvGJugzhl067LmYDMwyClH7gxgh3tT1S_lLpwE4PElUK1wOjCHKZpPkrLX-Vm8EnByXyFCmbwKu6WBvPhMVFLqUyPlnS-3--Dw4ET2KgVz6exZav0c-eQ4L9QKKXwmQJ9tIfyRMDia_m5JhCwajmAvQX13EpHAYVEK75MITl6jbg?key=IXpe4oV-OtSIbW3o2M5jrA)

### Evaluation of Propensity Score Matching Quality

- Compare the distribution of the treatment and matched control group
    
- Calculate the standardized mean difference of covariates, the rule of thumb value is 0.1
    

  

### Mediation Modeling

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfNwPaaf6vNESHe9xWmKt74bdXXt-Q7RUdGZCcBltJgHoJGPqVM-QpvZzWoIMm9vemUCqzza_YZ1D_nWazvEfwB-Q4yM0w1lsv80ki90zeu_qXXzf9XiP4jtlypOQPb0es8KbLqdIJQRoEU0jni_5ouFzPksqEgBceKv7iPU2R19cd4BwETvKw?key=IXpe4oV-OtSIbW3o2M5jrA)

Mediation modeling measures why a treatment is effective. This example estimates how much the decrease in support tickets is due to improved earnings understanding (measured by a survey question on a 1-5 scale) resulting from the earnings graph feature.

  

A conventional method that can be used is called [PROCESS](https://www.processmacro.org/index.html?uclick_id=124a5edd-67a2-4d26-a874-6f2c13d36042).  Uber has its own ML-based framework to do the analysis also.

  
  
  
**