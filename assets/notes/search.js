/*
 * search.js — client-side global search + scope filtering for the Paper Notes
 * Database (Requirement 14).
 *
 * Two clearly separated halves:
 *
 *   1. PURE, DOM-FREE logic (`matchQuery`, `filterPapers`) — a JavaScript mirror
 *      of the server-side search matching, exported so Vitest + fast-check can
 *      property-test them (design Testing Strategy → Property 9). Unchanged.
 *
 *   2. DOM glue (NOT exported) — powers two surfaces:
 *      a. A live dropdown under the Top_Bar search box on every page: as you
 *         type it shows a "会议" (conference) scope selector plus live results,
 *         restricting matches to the chosen conference.
 *      b. The full Search_Results_View page (/notes/search/), which reads the
 *         `q` and `conference` URL params, applies both, and offers the same
 *         conference selector.
 *      Guarded so it never runs under test (no `document`) and never throws.
 *
 * EMPTY-QUERY DECISION (Requirement 14.5): `matchQuery` returns false for an
 * empty/whitespace query, so an empty query yields the no-results state.
 */
(function (root, factory) {
  "use strict";
  var api = factory();

  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (typeof window !== "undefined") {
    window.NotesSearch = api;
  }
  if (typeof document !== "undefined") {
    api._attach(document);
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // --- Pure logic (exported, DOM-free) --------------------------------------

  function matchQuery(query, paper) {
    if (typeof query !== "string") return false;
    var needle = query.trim().toLowerCase();
    if (needle === "") return false;
    if (!paper || typeof paper !== "object") return false;

    var haystacks = [];
    if (typeof paper.title === "string") haystacks.push(paper.title);
    if (Array.isArray(paper.authors)) {
      for (var i = 0; i < paper.authors.length; i++) {
        if (typeof paper.authors[i] === "string") haystacks.push(paper.authors[i]);
      }
    }
    if (Array.isArray(paper.tags)) {
      for (var j = 0; j < paper.tags.length; j++) {
        if (typeof paper.tags[j] === "string") haystacks.push(paper.tags[j]);
      }
    }
    for (var k = 0; k < haystacks.length; k++) {
      if (haystacks[k].toLowerCase().indexOf(needle) !== -1) return true;
    }
    return false;
  }

  function filterPapers(query, papers) {
    if (!Array.isArray(papers)) return [];
    return papers.filter(function (paper) {
      return matchQuery(query, paper);
    });
  }

  // --- Shared DOM/index helpers (impure, not exported) ----------------------

  var SEARCH_INDEX_URL = "/notes/search-index.json";
  var _indexPromise = null;

  // Fetch and cache the search index once per page.
  function loadIndex(doc) {
    if (_indexPromise) return _indexPromise;
    var fetchFn = (doc.defaultView || window).fetch;
    if (typeof fetchFn !== "function") {
      return Promise.reject(new Error("fetch unavailable"));
    }
    _indexPromise = fetchFn(SEARCH_INDEX_URL).then(function (response) {
      if (!response || !response.ok) throw new Error("Failed to load search index");
      return response.json();
    });
    return _indexPromise;
  }

  // Scope dimensions: UI key -> index field.
  var SCOPE_FIELDS = { conference: "conference_label", track: "track" };

  // Restrict papers to a scope { conference, category, track } (empty = any).
  function applyScope(papers, scope) {
    var out = papers;
    Object.keys(SCOPE_FIELDS).forEach(function (key) {
      var value = scope[key];
      if (!value) return;
      var field = SCOPE_FIELDS[key];
      out = out.filter(function (p) {
        return p[field] === value;
      });
    });
    return out;
  }

  // Distinct values of an index field. Sorted ascending, or descending
  // (used for conference labels so the newest year appears first).
  function distinctValues(papers, field, descending) {
    var seen = {};
    var out = [];
    for (var i = 0; i < papers.length; i++) {
      var v = papers[i][field];
      if (v && !seen[v]) {
        seen[v] = true;
        out.push(v);
      }
    }
    out.sort(function (a, b) {
      return a < b ? -1 : a > b ? 1 : 0;
    });
    if (descending) out.reverse();
    return out;
  }

  function fillOptions(select, values) {
    if (!select) return;
    for (var i = 0; i < values.length; i++) {
      var opt = select.ownerDocument.createElement("option");
      opt.value = values[i];
      opt.textContent = values[i];
      select.appendChild(opt);
    }
  }

  // Populate the conference/track scope selects from the index.
  function populateScopeSelects(selects, papers) {
    fillOptions(selects.conference, distinctValues(papers, "conference_label", true));
    fillOptions(selects.track, distinctValues(papers, "track", false));
  }

  function readScope(selects) {
    return {
      conference: selects.conference ? selects.conference.value : "",
      track: selects.track ? selects.track.value : ""
    };
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

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
    for (var i = 0; i < papers.length; i++) html += renderListItem(papers[i]);
    container.innerHTML = '<div class="notes-list">' + html + "</div>";
  }

  function renderEmptyState(container, message) {
    container.innerHTML = '<div class="notes-empty">' + escapeHtml(message) + "</div>";
  }

  function resultsUrl(query, scope) {
    var url = "/notes/search/?q=" + encodeURIComponent(query);
    if (scope.conference) url += "&conference=" + encodeURIComponent(scope.conference);
    if (scope.category) url += "&category=" + encodeURIComponent(scope.category);
    if (scope.track) url += "&track=" + encodeURIComponent(scope.track);
    return url;
  }

  // --- Surface a: the Top_Bar live search dropdown --------------------------

  function attachDropdown(doc) {
    var input = doc.getElementById("notes-search-input");
    var panel = doc.getElementById("notes-search-panel");
    if (!input || !panel) return;

    var results = doc.getElementById("notes-search-panel-results");
    var selects = {
      conference: doc.getElementById("notes-search-scope-conference"),
      track: doc.getElementById("notes-search-scope-track")
    };
    var form = input.form;
    var LIMIT = 8;
    var populated = false;

    function ensureIndex() {
      return loadIndex(doc).then(function (papers) {
        if (!populated) {
          populateScopeSelects(selects, papers);
          populated = true;
        }
        return papers;
      });
    }

    function open() {
      panel.hidden = false;
      input.setAttribute("aria-expanded", "true");
    }
    function close() {
      panel.hidden = true;
      input.setAttribute("aria-expanded", "false");
    }

    function render(papers, query, scope) {
      if (!results) return;
      if (papers.length === 0) {
        results.innerHTML = '<div class="notes-search-panel__hint">未找到匹配的论文。</div>';
        return;
      }
      var shown = papers.slice(0, LIMIT);
      var html = '<div class="notes-list">';
      for (var i = 0; i < shown.length; i++) html += renderListItem(shown[i]);
      html += "</div>";
      if (papers.length > LIMIT) {
        html +=
          '<a class="notes-search-panel__more" href="' +
          escapeHtml(resultsUrl(query, scope)) +
          '">查看全部 ' + papers.length + " 个结果</a>";
      }
      results.innerHTML = html;
    }

    function update() {
      var query = input.value;
      if (query.trim() === "") {
        close();
        if (results) results.innerHTML = "";
        return;
      }
      ensureIndex()
        .then(function (papers) {
          var scope = readScope(selects);
          render(applyScope(filterPapers(query, papers), scope), query, scope);
          open();
        })
        .catch(function () {
          if (results) {
            results.innerHTML = '<div class="notes-search-panel__hint">搜索暂时不可用。</div>';
          }
          open();
        });
    }

    input.addEventListener("input", update);
    input.addEventListener("focus", function () {
      if (input.value.trim() !== "") update();
    });
    Object.keys(selects).forEach(function (key) {
      if (selects[key]) selects[key].addEventListener("change", update);
    });

    // Submit (Enter / button) -> full results page carrying q + scope.
    if (form) {
      form.addEventListener("submit", function (e) {
        var query = input.value;
        if (query.trim() === "") return;
        e.preventDefault();
        (doc.defaultView || window).location.href = resultsUrl(query, readScope(selects));
      });
    }

    // Close on outside click or Escape.
    doc.addEventListener("click", function (e) {
      if (panel.hidden) return;
      var withinForm = form && form.contains(e.target);
      if (!panel.contains(e.target) && !withinForm) close();
    });
    input.addEventListener("keydown", function (e) {
      if (e.key === "Escape") close();
    });
  }

  // --- Surface b: the full Search_Results_View page -------------------------

  function reflectQuery(doc, query) {
    var heading = doc.getElementById("notes-search-heading");
    if (heading) {
      heading.textContent = query && query.trim() !== "" ? 'Results for "' + query + '"' : "Search";
    }
    var input = doc.getElementById("notes-search-input");
    if (input && query) input.value = query;
  }

  function attachResultsPage(doc) {
    var container = doc.querySelector(".notes-search-results");
    if (!container) return;

    var selects = {
      conference: doc.getElementById("notes-search-results-conference"),
      track: doc.getElementById("notes-search-results-track")
    };
    var win = doc.defaultView || window;

    function readParams() {
      try {
        return new URLSearchParams(win.location.search || "");
      } catch (e) {
        return new URLSearchParams("");
      }
    }

    var params = readParams();
    var query = params.get("q") || "";
    var scope = {
      conference: params.get("conference") || "",
      track: params.get("track") || ""
    };
    reflectQuery(doc, query);

    function render(papers) {
      var matches = applyScope(filterPapers(query, papers), scope);
      if (matches.length === 0) renderEmptyState(container, "No papers found.");
      else renderResults(container, matches);
    }

    function syncUrl() {
      try {
        var url = new URL(win.location.href);
        Object.keys(SCOPE_FIELDS).forEach(function (key) {
          if (scope[key]) url.searchParams.set(key, scope[key]);
          else url.searchParams.delete(key);
        });
        win.history.replaceState(null, "", url);
      } catch (e) {
        /* history update is best-effort */
      }
    }

    if (query.trim() === "") renderEmptyState(container, "No papers found.");

    loadIndex(doc)
      .then(function (papers) {
        Object.keys(selects).forEach(function (key) {
          var select = selects[key];
          if (!select) return;
          var field = SCOPE_FIELDS[key];
          fillOptions(select, distinctValues(papers, field, key === "conference"));
          select.value = scope[key];
          select.addEventListener("change", function () {
            scope[key] = select.value;
            syncUrl();
            render(papers);
          });
        });
        if (query.trim() !== "") render(papers);
      })
      .catch(function () {
        if (query.trim() !== "") renderEmptyState(container, "Search is temporarily unavailable.");
      });
  }

  function attach(doc) {
    var run = function () {
      attachDropdown(doc);
      attachResultsPage(doc);
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
    _attach: attach
  };
});
