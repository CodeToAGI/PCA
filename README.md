# EP98 — PCA Explained (Principal Component Analysis)

**CodeToAGI · Machine Learning Series · Episode 25**  
**Section:** 05 – Unsupervised Learning

> You have 50 features. You can only see 2. Here's the fix.

---

## What this episode covers

- The curse of dimensionality (why you can't just plot 50 columns)
- Finding the direction of maximum variance — live visual
- Projecting onto PC1 vs a random axis
- Explained variance ratio & the scree plot
- PCA on a real 4+ feature dataset → clean 2D view
- PCA vs t-SNE vs feature selection
- Code: `StandardScaler` + `PCA` in scikit-learn
- Common mistakes (scaling, treating PCs as original features, etc.)

---

## Challenge Task

**Apply PCA to a High-Dimensional Dataset of Your Own**

1. Pick any dataset with **4+ numeric features** (Iris, Wine, or your own CSV).
2. **Standardize** the features first (`StandardScaler`) — don't skip this.
3. Fit PCA and print `explained_variance_ratio_` for each component.
4. Plot **cumulative** explained variance and pick `n_components` for **95%**.
5. Project onto **2 components** and scatter-plot colored by the true label.
6. Write one sentence: how much variance did you keep, and did the classes separate visually?

**Solution file:** [`ep98_challenge_solution.py`](ep98_challenge_solution.py)

```bash
pip install scikit-learn matplotlib numpy
python ep98_challenge_solution.py
