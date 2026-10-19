# Publishing this project on GitHub

The publication target is a separate private repository, `Gaptraf/orange-pi-6-plus-npu-investigation-clean`. The original private draft retains its old metadata and is not the publication source. Keep the clean copy private until the owner decides to publish it.

Use the GitHub-provided `noreply` author identity for every commit. A new commit does not remove private information from an older history. This project's clean local Git history starts from a new root commit; it does not inherit the original draft's commits.

Recommended repository name:

```text
orange-pi-6-plus-npu-investigation-clean
```

Suggested description:

```text
Orange Pi 6 Plus NPU investigation: correctness failures, runtime comparisons, custom GEMV, and the unresolved bandwidth bottleneck. Evidence and code.
```

## Publish the article as a repository

1. From the dashboard, choose **+ -> New repository** (or open [GitHub's new-repository form](https://github.com/new)).
2. Select owner `Gaptraf`, enter the name above, and choose **Public** when ready to publish.
3. If you will upload this complete prepared tree, leave automatic README, .gitignore, and license generation off to avoid conflicting copies. Create the repository.
4. Use the empty repository's **uploading an existing file** link, or **Add file -> Upload files** once it has content. Upload the contents of this project directory, preserving folders, then commit the files.
5. Confirm that `README.md` is at the repository root. GitHub will render it on the main repository page, with a link to the full `ARTICLE.md`.

For a larger folder upload, Git or GitHub Desktop is more reliable than dragging a ZIP into the web form. Uploading only the ZIP will not turn its internal README into the repository's landing page. The prepared tree contains more than 100 files, so a web upload may need multiple batches.

Using Git from **inside this prepared project directory**, with Git installed and authentication configured:

```sh
git init -b main
git add .
git commit -m "Publish Orange Pi 6 Plus NPU investigation and evidence"
git remote add origin https://github.com/Gaptraf/orange-pi-6-plus-npu-investigation-clean.git
git push -u origin main
```

Create the empty remote repository first. Git may ask you to configure your commit identity and authenticate; credentials do not belong in this project.

Expected repository URL after publication:

```text
https://github.com/Gaptraf/orange-pi-6-plus-npu-investigation-clean
```

Verify the visibility and contents of this clean destination before making it public.

## Publish the article as a GitHub Pages website

The ready static page is `docs/index.html`. It has its own stylesheet, no CDN dependency, no analytics, and no third-party scripts. `docs/.nojekyll` bypasses Jekyll processing.

After the files are pushed:

1. Open the repository's **Settings -> Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**.
3. Select branch **main**, folder **/docs**, and save.
4. Wait for deployment and use the **Visit site** link shown by GitHub.

Expected project-site URL after successful deployment:

```text
https://gaptraf.github.io/orange-pi-6-plus-npu-investigation-clean/
```

GitHub Pages is available for public repositories on GitHub Free. See [GitHub's site creation instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site) and [publishing-source configuration](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

The generated page points evidence and source links at the proposed GitHub repository. If you choose another owner or repository name, edit the constants in `analysis/build_site.py`, regenerate the page, and update the destinations in this guide.

## Make it visible from the profile

Once published, pin the repository on the profile and put its URL in the profile description or website field if desired. A separate profile README is optional; it requires a repository whose name matches the GitHub username. Neither a profile README nor GitHub Pages is required for readers to view the article in the project repository.

Suggested topics: `orange-pi`, `orange-pi-6-plus`, `cix`, `npu`, `zhouyi`, `llm`, `benchmark`, `hardware-review`.

Keep the measured scope visible when sharing: the investigation did not complete a full NPU LLM, and it did not prove that 24 GB/s is the physical limit. The recorded software failures and development experience stand on their own.
