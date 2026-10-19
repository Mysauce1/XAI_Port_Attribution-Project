# Publishing this project to GitHub

1. Create an empty GitHub repository (for example, `cruise-attribution-xai`). Do not initialize it with a README if you plan to push this folder as-is.
2. Extract this project ZIP and open a terminal in the extracted `cruise-attribution-xai` directory.
3. Review `data/processed/annotated_reviews.csv` and the generated `data/processed/review_splits.csv`; do not publish any data you do not have permission to share.
4. Initialize and push:

```bash
git init
git add .
git commit -m "Initial reproducible cruise destination attribution XAI project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

5. Open the repository's **Actions** tab and confirm the `tests` workflow passes.
6. After running BERT locally, keep model checkpoints out of Git (they are ignored). Commit code, configuration, results you intend to share, and a clear note of the model identifier/settings. If the dataset or derived annotations have sharing restrictions, replace them with an approved sample and document how authorized users can obtain the full data.

The Git remote URL above is a template; replace the username and repository with your actual GitHub values. No repository has been created or pushed on your behalf.
