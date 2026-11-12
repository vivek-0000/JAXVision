# Deploying JAXVision to Render (no Docker)

Result: one public URL that serves the dashboard and the API, the same app you see on localhost.

## Before you start

You need three things:

1. A free [GitHub](https://github.com) account.
2. A free [Render](https://render.com) account (sign up with GitHub, it is the easiest).
3. **A trained model in the `artifacts/` folder.** The server only shows real results if this folder is uploaded. Train on your computer or a TPU VM first:

   ```bash
   python -m jaxvision.train --dataset cifar10 --epochs 30
   ```

   When it finishes, `artifacts/` contains `checkpoint.msgpack`, `model_meta.json`, `metrics.json`, `benchmark.json` and `samples.npz`. All five are needed.

The folder `frontend/dist` (the built dashboard) is already included, so Render does not need Node. If you ever change the frontend code, rebuild it with `cd frontend && npm install && npm run build` before uploading.

## Step 1: Put the project on GitHub

**Without installing Git (easiest):**
1. On GitHub, click **New repository**, name it `jaxvision`, keep it public or private, and create it.
2. Click **uploading an existing file**.
3. Drag in the contents of the project folder: `jaxvision/`, `server/`, `frontend/` (without `node_modules`), `artifacts/`, `render.yaml`, `requirements-deploy.txt`, `requirements.txt` and the other root files. GitHub's web uploader accepts folders, up to 100 files per upload, so do it in two batches if needed.
4. Click **Commit changes**.

**With Git installed:**
```bash
git init && git add . && git commit -m "JAXVision"
git branch -M main
git remote add origin https://github.com/<your-username>/jaxvision.git
git push -u origin main
```
Check that `artifacts/` and `frontend/dist/` appear in the repo on GitHub. If they are missing, the site will load but show "No results yet".

## Step 2: Create the service on Render

1. In Render, click **New** and then **Blueprint**.
2. Connect your GitHub account and pick the `jaxvision` repo.
3. Render reads `render.yaml`, shows one web service called `jaxvision`, and asks you to confirm. Click **Apply**.
4. Wait for the first build, about 3 to 6 minutes, because it installs JAX. When the log says the service is live, open the URL at the top of the page.

(Prefer clicking through settings? Choose **New > Web Service** instead and enter: Runtime `Python`, Build command `pip install -r requirements-deploy.txt`, Start command `uvicorn server.app:app --host 0.0.0.0 --port $PORT`, plus environment variable `JAX_PLATFORMS=cpu`.)

## Step 3: Check it works

- `https://<your-app>.onrender.com/api/health` should show `"model_ready": true`.
- Open the main URL: the Overview should show your accuracy, and the Playground should classify the sample images.

## Good to know

- **Free plan sleeps.** After about 15 minutes without visitors the app goes to sleep, and the next visit takes up to a minute to wake it. Fine for a portfolio, but open the link yourself before showing it to someone.
- **Memory.** The app uses about 270 MB, and the free plan allows 512 MB. If Render ever reports it ran out of memory, move to the paid Starter plan or deploy to Hugging Face Spaces instead.
- **Updating.** Push or upload new files to GitHub and Render redeploys automatically. To publish a newly trained model, replace the `artifacts/` files the same way.
- **Training is not done on Render.** It only serves the model. Training stays on your computer or TPU.
- **Benchmarks** in `benchmark.json` reflect the machine you trained on, not Render.

## If something goes wrong

| What you see | Likely cause and fix |
| --- | --- |
| "No results yet" on every page | `artifacts/` was not uploaded to GitHub. Upload it and let Render redeploy. |
| Dashboard is blank or 404 on the main URL | `frontend/dist/` was not uploaded. Upload it. |
| Build fails while installing packages | Open the build log. Re-run the deploy. If it names a Python version, set the `PYTHON_VERSION` variable on the service to a 3.12 or 3.11 version Render lists. |
| "Out of memory" | Free plan limit. Use Starter or Hugging Face Spaces. |
| Page takes a minute the first time | The free instance was asleep. Wait and refresh. |
