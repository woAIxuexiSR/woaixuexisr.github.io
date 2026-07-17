# Authoring paper notes

This folder holds the content for the standalone **Paper Notes** section published at
`/notes/`. Each paper is one Markdown file. Jekyll scans this folder at build time and
generates the home page, per-conference pages, per-category pages, per-tag pages, and a
detail page for every note automatically. **To add a paper you only drop a new Markdown
file into the right folder and redeploy — no page code, template, CSS, or JavaScript
changes are needed.**

> This `README.md` is excluded from the Jekyll build (see `exclude:` in `_config.yml`), so
> it is a maintainer guide only and never becomes a published note.

## Folder naming (conference + year are auto-derived)

Notes live in **Conference_Folders** whose name encodes the conference and the year. You do
**not** put the conference or year inside a file — they are derived from the folder name:

| Folder name          | Derived conference | Derived year | Conference label     |
|----------------------|--------------------|--------------|----------------------|
| `SIGGRAPH<year>`     | `SIGGRAPH`         | `<year>`     | `SIGGRAPH <year>`    |
| `SIGGRAPHAsia<year>` | `SIGGRAPH Asia`    | `<year>`     | `SIGGRAPH Asia <year>` |

Examples: `SIGGRAPH2024/`, `SIGGRAPHAsia2024/`. A folder that does not match one of these
two patterns is ignored (its notes are skipped with a build warning).

An **empty** Conference_Folder is valid and still gets a card on the home page with a count
of zero. Because git and Jekyll ignore truly empty directories, keep an empty folder alive
with a `.gitkeep` file inside it (see `SIGGRAPH2023/`).

## File naming (kebab-case slug)

Name each file with a lowercase **kebab-case** slug and the `.md` extension. The slug is the
paper's unique id and becomes part of its URL (`/notes/paper/<slug>/`).

```
_notes/SIGGRAPHAsia2024/efficient-neural-path-guiding-with-4d-modeling.md
```

Good: `real-time-gaussian-splatting.md`  ·  Avoid: `Real Time Gaussian.md`, `paper_01.md`.

## Front matter schema

Every note begins with a YAML front-matter block delimited by `---`.

```yaml
---
title: "Efficient Neural Path Guiding with 4D Modeling"   # REQUIRED — string, full display title
authors:                                                   # REQUIRED — non-empty list
  - "Honghao Dong"
  - "Rui Su"
category: "Rendering"                                      # REQUIRED — exactly one of the 7 categories below
track: "Conference"                                        # REQUIRED — exactly "Conference" or "Journal"
institution: "Peking University"                           # required by CONVENTION — first-level institution(s); string or list
tags:                                                      # optional — list of free-form keywords
  - "path guiding"
  - "neural networks"
links:                                                     # paper required by CONVENTION; project/code optional
  paper: "https://doi.org/10.1145/3680528.3687687"         # required by CONVENTION — must be the ACM DOI link
  project: "https://woaixuexisr.github.io/papers/distributed-guiding/"
  code: "https://github.com/woAIxuexiSR/repo"
---
```

### Required fields (enforced by the build)

These are validated at build time; a note missing or getting any of them wrong is skipped with
a build warning (the build never fails, and other notes are unaffected):

- **`title`** — string, the full paper title used for display.
- **`authors`** — a non-empty list of author names.
- **`category`** — exactly **one** of the seven fixed categories (see below).
- **`track`** — exactly one of the two fixed values **`"Conference"`** or **`"Journal"`**.
  This is the Technical Papers track. It is orthogonal to the conference/year, is display-only
  (shown as a badge, not searchable or filterable), and is **not** derived from the folder name.

### Required by convention (author discipline, not build-enforced)

These are expected on every note as an authoring convention. The build does **not** currently
validate them, so a note missing them still publishes — but please always include them:

- **`institution`** — the **first-level institution(s)** only (e.g. `"Peking University"`,
  `"NVIDIA"`, `"MIT"`); not per-author, and no sub-departments. For a single-institution paper
  write a plain string; for a collaboration across institutions write a **list** — both forms
  are accepted and rendered joined with `; `:

  ```yaml
  institution: "Peking University"          # single institution
  # or, for a multi-institution collaboration:
  institution:
    - "Peking University"
    - "NVIDIA"
  ```
- **`links.paper`** — always provide the paper link, and it **must be the ACM DOI** URL in the
  form `https://doi.org/10.1145/<...>` (SIGGRAPH / SIGGRAPH Asia papers all have one).

### The seven fixed categories

Use one of these exact strings (spelling and capitalization must match):

1. `Rendering`
2. `Geometry & Modeling`
3. `Reconstruction`
4. `Animation & Simulation`
5. `Image & Video`
6. `Neural & Generative`
7. `HCI & XR`

### Optional fields

- **`tags`** — a list of free-form keywords. A paper may carry several. Each tag gets its own
  Tag page.
- **`links.project`** and **`links.code`** — optional URLs; include only the ones you have.
  Each renders as a link control on the detail page. (There is no `video` link.)

## Body: Markdown with LaTeX, no images

- The body (everything after the closing `---`) is the AI summary, written in **Markdown**.
- The body **may be empty** — a front-matter-only file is valid and its detail page shows an
  empty-state message in place of the note body.
- LaTeX math is supported and rendered by MathJax. **Always wrap math in double dollars
  `$$ ... $$`** — for BOTH inline and block math. Do **not** use single `$ ... $`: a single-`$`
  span is parsed as ordinary Markdown first, so underscores become italics and `[..](..)`
  becomes a link, corrupting the formula. With `$$ ... $$` the content is protected; math that
  sits inside a sentence renders inline, and math on its own line renders as a centered block.
- **Do not embed images** in the body.

## Workflow: drop a file + redeploy

1. Pick (or create) the correct `SIGGRAPH<year>/` or `SIGGRAPHAsia<year>/` folder.
2. Add a new `<kebab-case-slug>.md` file with valid front matter and a Markdown body.
3. Commit and push. The existing GitHub Pages workflow rebuilds the site.
4. The new paper automatically appears on the home count, its conference page, its category
   page, each of its tag pages, its own detail page, and the global search index — with no
   changes to templates, CSS, or JavaScript.
