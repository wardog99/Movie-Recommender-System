# Movie Recommendation System

This project explores movie recommendation with two approaches: item-to-item collaborative filtering using MovieLens ratings, and content-based filtering using movie metadata from the TMDB 5000 dataset. The work is organized as Jupyter notebooks for data preparation, similarity calculation, and example recommendations.

## Project Structure

```text
Recommender_System/
|-- README.md
|-- FindSimilarMovies.ipynb
|-- SimilarMoviesCollaborativeFiltering.ipynb
|-- tmdb_dataset_recommender.ipynb
`-- datasets/
	|-- ml-100k/
	|   |-- u.data
	|   |-- u.item
	|   `-- ...
	`-- archive/
		|-- tmdb_5000_movies.csv
		`-- tmdb_5000_credits.csv
```

## Notebooks

### `FindSimilarMovies.ipynb`

An exploratory MovieLens notebook. It loads user ratings and movie titles, builds a user-by-movie ratings table, examines movie rating counts, and uses rating correlations to find movies with similar user-rating patterns.

### `SimilarMoviesCollaborativeFiltering.ipynb`

A more structured MovieLens item-to-item recommender. It compares basic Pearson correlation with an overlap-aware score that shrinks correlations when fewer users rated both movies. Results include correlation, the number of common users, and total movie ratings.

### `tmdb_dataset_recommender.ipynb`

A content-based recommender. It combines movie overviews, genres, keywords, cast, and director into text tags, converts those tags to count vectors with `CountVectorizer`, and ranks movies using cosine similarity. The notebook's `recommend(movie)` example prints the five most similar titles.

## Datasets

- **MovieLens 100K**: `datasets/ml-100k/`. The recommender notebooks use `u.data` for user/movie ratings and `u.item` for movie IDs and titles. The 100K ratings dataset contains user IDs, movie IDs, ratings, and timestamps.
- **TMDB 5000 Movies and Credits**: `datasets/archive/`. The content-based notebook uses `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.

Dataset files are present in this workspace. Follow the dataset providers' terms when downloading, using, or redistributing them.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- pandas
- NumPy
- scikit-learn

Install the Python dependencies in your environment:

```bash
python -m pip install pandas numpy scikit-learn jupyter
```

Start Jupyter from the project directory:

```bash
jupyter notebook
```

Open the notebook you want to explore and run its cells from top to bottom.

## Dataset Paths

The TMDB notebook currently reads files from an absolute path under this project directory. For portability, change those paths to paths relative to the project root, such as:

```python
from pathlib import Path

DATA_DIR = Path("datasets/archive")
movies_path = DATA_DIR / "tmdb_5000_movies.csv"
credits_path = DATA_DIR / "tmdb_5000_credits.csv"
```

The MovieLens notebooks currently refer to `/Users/partha/AI_ML_Course/MLCourse/ml-100k`, which is outside this project. Update their dataset paths to `datasets/ml-100k` (or set a `Path` variable to that directory) before running them on another machine.

## How the Recommenders Work

### MovieLens: item-to-item collaborative filtering

The MovieLens notebooks compare rating vectors for pairs of movies. Movies rated similarly by users are treated as similar. The structured notebook also counts users who rated both movies and reduces the influence of correlations supported by small overlaps.

### TMDB: content-based filtering

The TMDB notebook represents each movie by words derived from its metadata. `CountVectorizer` creates a feature vector for each movie, and cosine similarity compares those vectors. This recommends movies with similar metadata, not necessarily movies that a particular user will like.

## Evaluation Notes

The current TMDB notebook has no user-level watch history, likes, or ratings. Its cosine similarity values measure metadata similarity; they are not recommendation accuracy or predicted user ratings. To evaluate user relevance, collect user interactions or use a held-out ratings dataset, then assess top-N recommendations with ranking metrics such as Precision@K, Recall@K, or NDCG@K.

The MovieLens notebooks use user ratings to calculate movie similarity, but do not yet implement a complete train/test evaluation workflow. A next step would be to hold out ratings, build similarities from training data only, and compare recommendations with the held-out interactions.

## Possible Next Steps

- Make all dataset paths relative to the project root.
- Add a held-out evaluation workflow and compare against a popularity baseline.
- Add a personalized collaborative-filtering model, such as biased matrix factorization, in a separate notebook.
- Return recommendation scores and titles as a table rather than printing titles only.
- Measure recommendation diversity and catalog coverage alongside relevance.
- Add a small user interface after the offline recommendation workflow is evaluated.
