/*
 * listing.js — in-page filtering + progressive "Show more" for the Paper Notes
 * listing pages (conference / category / tag).
 *
 * The stat controls double as filters:
 *   - the track row: 全部 (clear) / Conference / Journal
 *   - conference pages also render category chips
 * Clicking a chip filters the already-rendered rows in place (by their
 * data-category / data-track), so e.g. "Rendering" on the SIGGRAPH 2025 page
 * shows only that conference's Rendering papers — no navigation, no separate
 * cross-conference page.
 *
 * When no filter is active the list shows a preview (data-preview-limit) with a
 * "Show more" button that reveals more in batches. When a filter is active all
 * matching rows are shown and the button is hidden.
 *
 * Pure DOM glue, guarded so it is inert in a non-browser/test context.
 */
(function () {
  "use strict";
  if (typeof document === "undefined") return;

  var BATCH = 10;

  // Save the ordered (currently-filtered) list as the paper's sidebar context.
  function saveContext(label, items) {
    try {
      window.sessionStorage.setItem("notesNearbyCtx", JSON.stringify({ label: label, items: items }));
    } catch (e) {
      /* best-effort */
    }
  }

  function initListing(root) {
    var list = root.querySelector(".notes-list");
    if (!list) return;

    var rows = Array.prototype.slice.call(list.querySelectorAll(".notes-list-row"));
    // The server marks rows beyond the preview with `.is-extra` (hidden via CSS)
    // for the no-JS case. Once JS runs it fully controls visibility with inline
    // styles, so drop that class — otherwise the CSS rule would keep matching
    // "extra" rows hidden even when a filter wants to show them.
    rows.forEach(function (row) {
      row.classList.remove("is-extra");
    });
    var button = root.querySelector(".notes-showmore");
    var emptyMsg = root.querySelector(".notes-list-empty");
    var chips = Array.prototype.slice.call(root.querySelectorAll("[data-filter]"));
    var preview = parseInt(list.getAttribute("data-preview-limit"), 10) || 10;

    var state = { category: null, track: null, revealed: preview };

    function filtering() {
      return state.category !== null || state.track !== null;
    }

    function matches(row) {
      if (state.category && row.getAttribute("data-category") !== state.category) return false;
      if (state.track && row.getAttribute("data-track") !== state.track) return false;
      return true;
    }

    function apply() {
      var matchingTotal = 0;
      var shown = 0;

      rows.forEach(function (row) {
        if (!matches(row)) {
          row.style.display = "none";
          return;
        }
        matchingTotal++;
        if (filtering()) {
          row.style.display = ""; // show all matches when filtering
          shown++;
        } else if (shown < state.revealed) {
          row.style.display = "";
          shown++;
        } else {
          row.style.display = "none";
        }
      });

      if (button) {
        if (!filtering() && state.revealed < matchingTotal) {
          button.style.display = "";
          button.textContent = "Show more (" + (matchingTotal - state.revealed) + " more)";
        } else {
          button.style.display = "none";
        }
      }

      if (emptyMsg) emptyMsg.hidden = matchingTotal !== 0;

      chips.forEach(function (chip) {
        var dim = chip.getAttribute("data-filter");
        var value = chip.getAttribute("data-value");
        var active =
          (dim === "all" && state.track === null) ||
          (dim === "category" && state.category === value) ||
          (dim === "track" && state.track === value);
        chip.classList.toggle("is-active", active);
      });
    }

    chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        var dim = chip.getAttribute("data-filter");
        var value = chip.getAttribute("data-value");
        if (dim === "all") {
          state.track = null; // "全部" clears only the track dimension
        } else if (dim === "category") {
          state.category = state.category === value ? null : value;
        } else if (dim === "track") {
          state.track = state.track === value ? null : value;
        }
        state.revealed = preview; // reset pagination when the filter changes
        apply();
      });
    });

    if (button) {
      button.addEventListener("click", function () {
        state.revealed += BATCH;
        apply();
      });
    }

    // Clicking a paper carries the current (filtered) list as its sidebar context.
    var headingEl = root.querySelector(".notes-page-heading__title");
    list.addEventListener("click", function (e) {
      var link = e.target.closest ? e.target.closest(".notes-list-item") : null;
      if (!link) return;
      var items = [];
      rows.forEach(function (row) {
        if (!matches(row)) return;
        var a = row.querySelector(".notes-list-item");
        if (!a) return;
        var titleEl = row.querySelector(".notes-list-item__title");
        items.push({
          title: (titleEl ? titleEl.textContent : a.textContent).trim(),
          url: a.getAttribute("href")
        });
      });
      var parts = [headingEl ? headingEl.textContent.trim() : "列表"];
      if (state.category) parts.push(state.category);
      if (state.track) parts.push(state.track);
      saveContext(parts.join(" · "), items);
    });

    apply();
  }

  function init() {
    initListing(document);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
