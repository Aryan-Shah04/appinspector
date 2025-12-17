# AppInspector AI

## 1. Quick Start (Local)

1. **Unzip** this folder.
2. Open terminal in this folder.
3. Run:
   ```bash
   npm install
   npm run dev
   ```

## 2. How to Host on GitHub Pages (Free)

1. **Create a Repository** on GitHub.
2. **Push your code**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```
3. **Deploy**:
   ```bash
   npm run deploy
   ```

## Troubleshooting "ENAMETOOLONG" Error
If `npm run deploy` fails with an error like `spawn ENAMETOOLONG`, it means Windows path limits are blocking the automatic script.

**Solution (Manual Deploy):**
Run these commands in your terminal one by one:
```bash
git checkout -b gh-pages
npm run build
git add dist -f
git commit -m "Manual deploy"
git subtree push --prefix dist origin gh-pages
```
Then switch back to your main branch:
```bash
git checkout main
```
