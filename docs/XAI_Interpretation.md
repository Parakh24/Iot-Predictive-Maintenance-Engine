# XAI Module – SHAP Interpretations for XGBoost Model

## 1. Top 5 Feature Insights
| Feature      | Impact on Prediction           | Interpretation                     |
|-------------|-------------------------------|-----------------------------------|
| Temperature | High → increases failure risk | Critical driver; monitor above 50°C |
| Vibration   | High → increases failure risk | Strong positive influence on failure |
| Pressure    | Low → increases failure risk  | Inversely related; low values increase risk |
| Voltage     | High → slight increase risk   | Minor effect, keep within normal range |
| Humidity    | Minimal effect                | Not significant for predictions    |

---

## 2. Introduction
This document explains the predictions of the XGBoost predictive maintenance model using SHAP (SHapley Additive exPlanations). The goal is to provide engineering-friendly insights into feature contributions and model behavior.

---

## 3. SHAP Summary Plot
![SHAP Summary](../explain/shap_outputs/shap_summary_plot.png)

**Interpretation:**
- Temperature and vibration are the most important features.  
- Higher temperature → higher failure probability.  
- High vibration → increases risk.  
- Pressure has slight negative effect (lower values increase failure).  
- Voltage and humidity have minor influence.

---

## 4. SHAP Bar Plot
![SHAP Bar Plot](../explain/shap_outputs/shap_bar_plot.png)

**Interpretation:**
- Shows the **overall average impact** of each feature on model output.  
- Confirms that temperature and vibration dominate model predictions.  
- Helps quickly identify key features engineers should monitor.

---

## 5. SHAP Force Plot
![SHAP Force Plot](../explain/shap_outputs/shap_force_plot_instance_0.png)

**Interpretation:**
- Force plot visualizes contributions for a single instance.  
- Red arrows/features increase predicted failure probability.  
- Blue arrows/features reduce predicted failure probability.  
- Example: High vibration pushes prediction higher, low pressure also increases risk.

---

## 6. SHAP Decision Plot
![SHAP Decision Plot](../explain/shap_outputs/shap_decision_plot.png)

**Interpretation:**
- Decision plot shows **how features cumulatively affect prediction**.  
- Each line represents a sample moving from base value → final prediction.  
- Helps engineers understand feature impact across multiple predictions.

---

## 7. SHAP Waterfall Plot
![SHAP Waterfall](../explain/shap_outputs/shap_waterfall_plot_instance_0.png)

**Interpretation:**
- Waterfall plot explains **individual prediction step by step**.  
- Shows base value → feature contributions → final prediction.  
- Engineers can see exactly which features increased or decreased risk for a sample.

---

## 8. How to Use the XAI Module
- Run `src/explain/shap_explain.py` (if needed for new predictions).  
- SHAP plots will be saved in `shap_output/`.  
- Engineers can reference these plots for **model explainability** and **feature impact understanding**.
