# multilevelmodel

Multi Level modeling with sklearn regressors

A Python wrapper for creating **Multi-Level Models** (also known as hierarchical or stratified models) using standard `scikit-learn` regressors.

This library allows you to automatically fit independent regression models for specific subgroups of your data (e.g., different regions, time periods, or categories) while maintaining a unified interface for fitting, predicting, and visualizing results.

## 🚀 Key Features

* **Group-Based Fitting**: Automatically partitions data based on specified columns (e.g., `['Region', 'Month']`) and fits a separate model for each group.
* **Scikit-Learn Compatible**: Works with any sklearn-like regressor (LinearRegression, Ridge, Lasso, etc.) as the base estimator.
* **Smart Fallback**: Supports custom strategies to handle missing groups during prediction (e.g., "if specific group is missing, use the average model").
* **Constant Value Handling**: Automatically detects if `X` is constant within a group and switches to a robust `InterceptOnlyRegressor` (predicts mean `y`).
* **Built-in Visualization**: Includes powerful plotting tools (`plot_profile`, `plot_facet_grid`) to visualize model behavior across different groups.
* **Persistence**: Easy save/load functionality using `pickle`.

---

## 📦 Installation

This is a standalone Python module. Simply download the `model.py` file and include it in your project directory.

```python
from multilevelmodel.model import MultiLevelModel