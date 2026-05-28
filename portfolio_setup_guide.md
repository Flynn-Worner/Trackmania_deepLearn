# Project Portfolio Setup Guide

This guide will walk you through setting up your project repository and GitHub Pages portfolio, based on the video instructions.

## 1. Create Your GitHub Repository
1. Go to GitHub and click **New Repository**.
2. Name your repository (e.g., `41118_project_sample`).
3. Set the repository visibility to **Public**.
4. **Add a `.gitignore` file**: Select a template relevant to your project (e.g., Python). This prevents you from accidentally uploading configuration files or large caches.
5. **Add a License**: The MIT license is a good default choice, though optional.
6. Click **Create repository**.

## 2. Set Up Your Local Workspace
1. Copy the clone URL from your new repository.
2. Open your terminal and clone the repository:
   ```bash
   git clone <your-repository-url>
   ```
3. Navigate into your cloned folder and set up your project structure. Example structure:
   - `data/`
   - `train.py`
   - `test.py`
   - `figures/`
4. Commit and push your initial structure:
   ```bash
   git add .
   git commit -m "Initial commit with project structure"
   git push origin main
   ```

## 3. Create the GitHub Pages Branch
To host your portfolio website, you will create a dedicated branch.
1. In GitHub (or locally), create a new branch from `main` (e.g., `gh-pages` or `page`).
2. Pull the branch to your local machine and switch to it:
   ```bash
   git checkout <branch-name>
   ```
3. Remove the raw code/data files that you don't need on the website branch (keep files like `README.md` and `LICENSE`). 
4. Copy your static website template files (like `index.html`) into this branch.
5. Add, commit, and push the website files to GitHub.

## 4. Set Up the `README.md`
Your `README.md` file should act as the primary documentation for your repository. Use Markdown to format it (e.g., `#` for main headings, `##` for subheadings). 

**Required `README.md` Content:**
- **Short Description**: A brief summary of your project.
- **Group Information**: Your group number and the names of all team members.
- **Installation Instructions**: What dependencies are required to run the code.
- **Run Commands**: Explicit commands on how to run your code (e.g., `python test.py`).
- **Expected Output**: What should happen when the code is run.

## 5. Enable GitHub Pages
1. On GitHub, go to your repository **Settings**.
2. In the left sidebar, click on **Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the branch you created for your website (e.g., `gh-pages`) and click **Save**.
5. GitHub will generate a link to your hosted portfolio. 
   - *Tip: Add this link to the "About" section on your repository's main page so it is easy to find.*

## 6. Customize Your Portfolio Website (`index.html`)
You will need to edit your `index.html` file to reflect your specific project. You can do this locally or directly through the GitHub web editor.

**Things to update in `index.html`:**
- **Page Title & Headings**: Change the default template titles to your project name.
- **Team Members**: Add your specific group composition.
- **Abstract/Summary**: Write an overview of your AI/project work.
- **Media**: Add your own images (e.g., AI-generated teasers) and embed your project video (via YouTube or GitHub).
- **Results & Discussion**: Detail your findings and project outcomes.
- **Links**: Update or remove any placeholder template links (e.g., links to the original creator's profile) and replace them with your own.

> [!NOTE]  
> **Caching Delay:** When you make changes to your `index.html` and push them to GitHub, it can take up to 15 minutes for the live website to update due to browser and server caching. If you don't see your changes immediately, grab a coffee and check back later!

---
*Make sure you fill out all required sections to satisfy your marking criteria. Good luck with your project!*
