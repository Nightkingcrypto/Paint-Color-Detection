# Using DVC in this project (for your report)

You can **explain** and optionally **demonstrate** how DVC would be used to
version large datasets and model artefacts.

Example steps (run in the project root):

```bash
# 1. Initialise DVC in the repo
dvc init

# 2. Tell DVC to track the dataset folder (instead of Git)
dvc add data/raw

# This creates data/raw.dvc which is a small text metafile.
# You commit this file to Git, but NOT the raw data.

git add data/raw.dvc .gitignore
git commit -m "Track dataset with DVC"

# 3. Configure a remote (for example, Google Drive, S3, or a shared folder)
dvc remote add -d myremote <remote-url>
dvc push    # uploads data to remote storage
```

In your **Jupyter notebook report**, you can write that:

- **Git** tracks the code, configs, and small meta files.
- **DVC** tracks large data and model weights.
- Team members can run `dvc pull` to reproduce the same dataset and models.

For this coursework it's enough to show commands and explain the idea, even if you
don't fully configure a cloud remote.
