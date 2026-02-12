This folder contains the 2 end to end notebook built for the project.

Each notebook runs an end to end workflow including data ingestion, cleaning, feature engineering, multiple model fits and evaluation.

The two notebooks differ only in how TF‑IDF features are constructed.

- Notebook 1 builds a single TF‑IDF feature matrix by combining all free‑text fields into one unified text input.

- Notebook 2 builds separate TF‑IDF feature matrices for each free‑text field (e.g., description, requirements, benefits) and then uses them as distinct feature blocks in the model.
