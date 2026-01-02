"""
Created on Tue Nov 18 18:06:02 2025

author: @adisarno
description: Multi Level modeling with sklearn regressors
    
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

class InterceptOnlyRegressor:
    """Regressor that always predicts the mean (slope=0)."""
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.intercept_ = float(np.average(y))
        else:
            self.intercept_ = float(np.average(y, weights=sample_weight))
        self.coef_ = np.array([0.0])
        return self

    def predict(self, X):
        return np.full(len(X), self.intercept_)
    
    def __repr__(self):
        return f"<InterceptOnlyRegressor intercept={getattr(self, 'intercept_', None):.4f}>"

class MultiLevelModel:
    def __init__(self, 
                 groups,  
                 x_col,  
                 target,
                 base_regressor=LinearRegression(fit_intercept=True),
                 min_obs=1, 
                 enable_cache=True,
                 fallback_strategy=None): # NEW: Function for custom fallback logic
        """
        fallback_strategy : callable, optional
            A function that accepts (key_tuple, groups_list) and returns a
            new alternative key_tuple to try if the main one is missing.
        """
        self.groups = groups
        self.x_col = x_col
        self.target = target
        self.base_regressor = base_regressor
        self.min_obs = min_obs
        self.enable_cache = enable_cache
        self.fallback_strategy = fallback_strategy
        
        self.models = {}      # Map: KeyTuple -> sklearn model
        self._cache = {}      # Cache for key resolution

    def fit(self, df: pd.DataFrame, sample_weights=None):
        assert all(g in df.columns for g in self.groups), f"Groups {self.groups} missing in DataFrame"
        assert self.x_col in df.columns, "x_col missing"
        assert self.target in df.columns, "target missing"

        self.models = {}
        self._cache = {}

        # Generic grouping
        # If groups has 1 element, groupby returns single values, force to tuple
        is_single_group = len(self.groups) == 1
        
        for key, g in df.groupby(self.groups):
            if is_single_group: key = (key,) # Always normalize to tuple

            if len(g) < self.min_obs:
                continue

            x = g[self.x_col].values
            y = g[self.target].values

            # Generic logic: If X is constant in the group, slope = 0
            # (This implicitly replaces the old "Weekend" logic if weekend data had constant X)
            if np.allclose(x, x[0]):
                reg = InterceptOnlyRegressor().fit(None, y)
                self.models[key] = reg
                continue

            # Standard model fit
            Xmat = x.reshape(-1, 1)
            reg = copy.deepcopy(self.base_regressor)

            if sample_weights is not None:
                # Weight alignment via index
                w = sample_weights.loc[g.index].values
                reg.fit(Xmat, y, sample_weight=w)
            else:
                reg.fit(Xmat, y)

            self.models[key] = reg

        return self

    def _get_model(self, key_tuple):
        # 1. Check Cache
        if self.enable_cache and key_tuple in self._cache:
            return self._cache[key_tuple], key_tuple

        # 2. Check Direct Match
        if key_tuple in self.models:
            if self.enable_cache: self._cache[key_tuple] = self.models[key_tuple]
            return self.models[key_tuple], key_tuple

        # 3. Check Fallback Strategy (Injected externally, not hardcoded)
        if self.fallback_strategy is not None:
            alt_key = self.fallback_strategy(key_tuple, self.groups)
            if alt_key and alt_key in self.models:
                if self.enable_cache: self._cache[key_tuple] = self.models[alt_key]
                return self.models[alt_key], alt_key

        # 4. No model found
        if self.enable_cache: self._cache[key_tuple] = None
        return None, key_tuple

    def predict_row(self, row):
        # Dynamic key construction based on self.groups
        key_tuple = tuple(row[g] for g in self.groups)
        
        model, _ = self._get_model(key_tuple)
        if model is None:
            return np.nan

        x = row[self.x_col]

        if isinstance(model, InterceptOnlyRegressor):
            Xmat = np.array([[0.0]])
        else:
            Xmat = np.array([[x]])

        return model.predict(Xmat)[0]

    def predict(self, df: pd.DataFrame):
        # Note: iterrows is slow, but keeping original logic for consistency
        return np.array([self.predict_row(row) for _, row in df.iterrows()])

    def explain_group(self, filter_dict):
        """
        Example usage: model.explain_group({'Month': 1, 'Hour': 10, ...})
        """
        # Order values based on self.groups order
        try:
            key_tuple = tuple(filter_dict[g] for g in self.groups)
        except KeyError as e:
            print(f"Missing key in filter_dict: {e}")
            return

        print("Requested key:", key_tuple)
        model, used_key = self._get_model(key_tuple)

        if model is None:
            print(" → NO MODEL FOUND")
            return

        print(" → Model found. Used key:", used_key)
        if hasattr(model, "coef_"):
            print("Coefficients:", model.coef_)
        if hasattr(model, "intercept_"):
            print("Intercept:", model.intercept_)

    # --- GENERALIZED PLOTTING ---
    
    def plot_profile(self, 
                     profile_axis_col, # The column to place on the X-axis (e.g., "Hour")
                     filters,          # Dictionary {Column: Value} to fix other groups
                     x_col_values=[0.5, 1.0, 1.5], 
                     ax=None, 
                     show_legend=True):
        
        if ax is None:
            plt.figure(figsize=(10, 5))
            ax = plt.gca()

        # 1. Identify model keys that satisfy the filters
        # Filter column indices
        filter_indices = {self.groups.index(k): v for k, v in filters.items()}
        profile_idx = self.groups.index(profile_axis_col)

        relevant_keys = []
        for key in self.models.keys():
            match = True
            for idx, val in filter_indices.items():
                if key[idx] != val:
                    match = False
                    break
            if match:
                relevant_keys.append(key)
        
        # Sort keys based on profile axis (e.g., Hour)
        relevant_keys.sort(key=lambda x: x[profile_idx])
        
        if not relevant_keys:
            if ax:
                ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
            return

        x_axis_values = [k[profile_idx] for k in relevant_keys]

        # 2. Generate predictions
        # For generality, reconstruct a dummy mini-dataframe to use predict_row
        # or call the saved model directly (faster)
        
        for sim_x in x_col_values:
            y_preds = []
            for key in relevant_keys:
                model = self.models[key]
                if isinstance(model, InterceptOnlyRegressor):
                    pred = model.predict([[0]])[0]
                else:
                    pred = model.predict([[sim_x]])[0]
                y_preds.append(pred)
            
            ax.plot(x_axis_values, y_preds, marker='.', label=f"{self.x_col}={sim_x}")

        ax.set_xlabel(profile_axis_col)
        ax.set_ylabel(self.target)
        title_parts = [f"{k}={v}" for k,v in filters.items()]
        ax.set_title(" | ".join(title_parts), fontsize=9)
        if show_legend:
            ax.legend()
    
    def plot_facet_grid(self, 
                        grid_row_group,
                        grid_col_groups,    
                        x_axis_col,         
                        x_col_values=[0.5, 0.8, 1.0, 1.2, 1.5],
                        figsize=None, 
                        save_path=None):
        """
        Generates a grid of plots.
        - grid_row_group: The group variable that defines the rows of the grid.
        - grid_col_groups: The group variable(s) that define the columns of the grid.
        - x_axis_col: The variable to plot on the X-axis of each subplot.
        """
        
        # 1. Normalize grid_col_groups to a list if it's a single string
        if isinstance(grid_col_groups, str):
            grid_col_groups = [grid_col_groups]

        # 2. Identify indices in the main group tuple
        try:
            row_idx = self.groups.index(grid_row_group)
            col_indices = [self.groups.index(g) for g in grid_col_groups]
        except ValueError as e:
            print(f"Error: Group not found in model structure. {e}")
            return

        # 3. Extract unique values present in the trained model
        # Unique values for grid rows
        unique_row_vals = sorted(list(set(k[row_idx] for k in self.models.keys())))
        
        # Unique values for grid columns (as tuples if multiple groups are used)
        unique_col_vals = sorted(list(set(
            tuple(k[i] for i in col_indices) for k in self.models.keys()
        )))

        n_rows = len(unique_row_vals)
        n_cols = len(unique_col_vals)

        if n_rows == 0 or n_cols == 0:
            print("Model empty or no grouping columns found.")
            return

        print(f"Generating grid: {n_rows} Rows ({grid_row_group}) x {n_cols} Cols ({grid_col_groups})...")

        # 4. Create Subplots
        if not figsize:
            figsize = (4 * n_cols, 3 * n_rows)
            
        fig, axes = plt.subplots(n_rows, n_cols, 
                                 figsize=figsize, 
                                 sharex=True, sharey=True)

        # Handle edge cases where axes is not a 2D array (1 row or 1 col)
        if n_rows == 1 and n_cols == 1: 
            axes = np.array([[axes]])
        elif n_rows == 1: 
            axes = axes.reshape(1, -1)
        elif n_cols == 1: 
            axes = axes.reshape(-1, 1)

        # 5. Populate Grid
        for i, r_val in enumerate(unique_row_vals):
            for j, c_vals in enumerate(unique_col_vals):
                ax = axes[i][j]
                
                # Show legend only in the top-right plot to avoid clutter
                show_leg = (i == 0 and j == n_cols - 1)
                
                # Construct the filter dictionary for this specific subplot
                # Start with the row filter
                current_filters = {grid_row_group: r_val}
                
                # Add the column filters (handle single value vs tuple logic)
                # If c_vals is a scalar (single column group), wrap in tuple for zip
                if len(grid_col_groups) == 1 and not isinstance(c_vals, tuple):
                    c_vals = (c_vals,)
                    
                for col_name, col_val in zip(grid_col_groups, c_vals):
                    current_filters[col_name] = col_val

                # Reuse the logic of plot_profile
                self.plot_profile(
                    profile_axis_col=x_axis_col,
                    filters=current_filters,
                    x_col_values=x_col_values,
                    ax=ax,
                    show_legend=show_leg
                )
                
                # Axis Labels (Only on outer edges)
                if i == n_rows - 1: 
                    ax.set_xlabel(x_axis_col)
                if j == 0: 
                    ax.set_ylabel(f"{grid_row_group}={r_val}\n{self.target}")
                
                if i == 0:
                    title_str = " | ".join([f"{n}={v}" for n, v in zip(grid_col_groups, c_vals)])
                    ax.set_title(title_str, fontsize=9)

        plt.tight_layout()
        
        if save_path:
            try:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved in: {save_path}")
            except Exception as e:
                print(f"Error while saving in {save_path}: {e}")
        
        plt.show()
            
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def __repr__(self):
        if self.groups and self.target is not None:
            return f"<MultiLevelModel(groups = {self.groups}, x_col = {self.x_col}, target = {self.target}))>"
        else:
            return "<MultiLevelModel>"