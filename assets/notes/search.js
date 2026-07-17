/*
 * search.js — client-side global search for the Paper Notes Database
 * (Requirement 14).
 *
 * This file is split into two clearly separated halves:
 *
 *   1. PURE, DOM-FREE logic (`matchQuery`, `filterPapers`) — a JavaScript mirror
 *      of the server-side search matching. These functions have no DOM, network,
 *      or Jekyll dependency and are exported so Vitest + fast-check can property-
 *      test them in isolation (design Testing Strategy → Property 9).
 *
 *   2. DOM glue (NOT exported, NOT pure) — reads the `q` query parameter, fetches
 *      `/notes/search-index.json` once, and renders the matching papers into the
 *      `.notes-search-results` container of the Search_Results_View. It is guarded
 *      so it never runs under test (no `document`) and never throws uncaught.
 *
 * Module format: a UMD-ish wrapper. The pure API is attached to
 * `window.NotesSearch` for the browser (the base layout loads this as a classic
 * `<script defer>`), AND exported via `module.exports` so Vitest can
 * `import { matchQuery, filterPapers } from '.../search.js'`.
 *
 * EMPTY-QUERY DECISION (Requirement 14.5): `matchQuery` returns `false` for an
 * empty or whitespace-only query, so `filterPapers` returns an empty array for
 * such a query. The render layer therefore shows the "no papers found" empty
 * state for a submitted empty query rather than listing every paper — matching
 * Requirement 14.5 ("a submitted empty query yields the no-results/empty state").
 */
(function (root, factory) {
  "use strict";
  var api = factory();

  // CommonJS / Vitest (Node) — export the pure API for testing.
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }

  // Browser — expose the pure API on the global for any inline consumer.
  if (typeof window !== "undefined") {
    window.NotesSearch = api;
  }

  // DOM glue: only run in a real browser document that actually contains the
  // Search_Results_View. Under Vitest (Node, no `document`) this block is
  // skipped entirely, keeping the import side-effect free.
  if (typeof document !== "undefined") {
    api._attach(document);
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // --- Pure logic (exported, DOM-free) --------------------------------------

  /**
   * Case-insensitive substring match of a query against one paper.
   *
   * Returns true iff the trimmed query occurs (case-insensitively) as a
   * substring of the paper's title, any author name, or any tag. An empty or
   * whitespace-only query returns false (see EMPTY-QUERY DECISION above).
   *
   * @param {string} query
   * @param {{title?:string, authors?:string[], tags?:string[]}} paper
   * @returns {boolean}
   */
  function matchQuery(query, paper) {
    if (typeof query !== "string") {
      return false;
    }
    var needle = query.trim().toLowerCase();
    if (needle === "") {
      return false;
    }
    if (!paper || typeof paper !== "object") {
      return false;
    }

    var haystacks = [];
    if (typeof paper.title === "string") {
      haystacks.push(paper.title);
    }
    if (Array.isArray(paper.authors)) {
      for (var i = 0; i < paper.authors.length; i++) {
        if (typeof paper.authors[i] === "string") {
          haystacks.push(paper.authors[i]);
        }
      }
    }
    if (Array.isArray(paper.tags)) {
      for (var j = 0; j < paper.tags.length; j++) {
        if (typeof paper.tags[j] === "string") {
          haystacks.push(paper.tags[j]);
        }
      }
    }

    for (var k = 0; k < haystacks.length; k++) {
      if (haystacks[k].toLowerCase().indexOf(needle) !== -1) {
        return true;
      }
    }
    return false;
  }

  /**
   * Filter a list of papers to the subset matching the query.
   *
   * @param {string} query
   * @param {Array} papers
   * @returns {Array} the matched subset (empty array for an empty query)
   */
  function filterPapers(query, papers) {
    if (!Array.isArray(papers)) {
      return [];
    }
    return papers.filter(function (paper) {
      return matchQuery(query, paper);
    });
  }

  // --- DOM glue (impure, not exported) --------------------------------------

  var SEARCH_INDEX_URL = "/notes/search-index.json";

  // Read the `q` parameter from the current URL query string.
  function readQuery(doc) {
    try {
      var search = (doc.defaultView || window).location.search || "";
      return new URLSearchParams(search).get("q") || "";
    } catch (e) {
      return "";
    }
  }

  // Escape text for safe insertion into HTML.
  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Render one search-result List_Item. Mirrors the server-side List_Item
  // (title, authors, conference_label, tags) but omits the Track badge because
  // the search index does not carry `track`. The whole item links to paper.url.
  function renderListItem(paper) {
    var authors = Array.isArray(paper.authors) ? paper.authors : [];
    var tags = Array.isArray(paper.tags) ? paper.tags : [];
    var href = typeof paper.url === "string" ? paper.url : "";

    var html = '<a class="notes-list-item" href="' + escapeHtml(href) + '">';
    html += '<h3 class="notes-list-item__title">' + escapeHtml(paper.title) + "</h3>";
    if (authors.length > 0) {
      html += '<p class="notes-list-item__authors">' + escapeHtml(authors.join(", ")) + "</p>";
    }
    html += '<div class="notes-list-item__meta">';
    if (paper.conference_label) {
      html += '<span class="notes-list-item__conference">' + escapeHtml(paper.conference_label) + "</span>";
    }
    if (tags.length > 0) {
      html += '<span class="notes-tags">';
      for (var i = 0; i < tags.length; i++) {
        html += '<span class="notes-badge notes-tag">' + escapeHtml(tags[i]) + "</span>";
      }
      html += "</span>";
    }
    html += "</div></a>";
    return html;
  }

  function renderResults(container, papers) {
    var html = "";
    for (var i = 0; i < papers.length; i++) {
      html += renderListItem(papers[i]);
    }
    container.innerHTML = '<div class="notes-list">' + html + "</div>";
  }

  function renderEmptyState(container, message) {
    container.innerHTML = '<div class="notes-empty">' + escapeHtml(message) + "</div>";
  }

  // Optional: reflect the query in the heading and prime the top-bar input.
  function reflectQuery(doc, query) {
    var heading = doc.getElementById("notes-search-heading");
    if (heading) {
      if (query && query.trim() !== "") {
        heading.textContent = 'Results for "' + query + '"';
      } else {
        heading.textContent = "Search";
      }
    }
    var input = doc.getElementById("notes-search-input");
    if (input && query) {
      input.value = query;
    }
  }

  // Wire up the Search_Results_View. Runs only when the container is present.
  function attach(doc) {
    var run = function () {
      var container = doc.querySelector(".notes-search-results");
      if (!container) {
        return; // Not the search page — nothing to do.
      }

      var query = readQuery(doc);
      reflectQuery(doc, query);

      // Empty/whitespace-only query -> no-results empty state (Requirement 14.5).
      if (query.trim() === "") {
        renderEmptyState(container, "No papers found.");
        return;
      }

      var fetchFn = (doc.defaultView || window).fetch;
      if (typeof fetchFn !== "function") {
        renderEmptyState(container, "Search is temporarily unavailable.");
        return;
      }

      fetchFn(SEARCH_INDEX_URL)
        .then(function (response) {
          if (!response || !response.ok) {
            throw new Error("Failed to load search index");
          }
          return response.json();
        })
        .then(function (papers) {
          var matches = filterPapers(query, papers);
          if (matches.length === 0) {
            renderEmptyState(container, "No papers found.");
          } else {
            renderResults(container, matches);
          }
        })
        .catch(function () {
          // Non-blocking failure message; the rest of the page stays usable.
          renderEmptyState(container, "Search is temporarily unavailable.");
        });
    };

    if (doc.readyState === "loading") {
      doc.addEventListener("DOMContentLoaded", run);
    } else {
      run();
    }
  }

  return {
    matchQuery: matchQuery,
    filterPapers: filterPapers,
    // Exposed for the UMD wrapper to invoke browser-only DOM wiring.
    _attach: attach
  };
});
