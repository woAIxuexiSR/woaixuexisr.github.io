/*
 * sidebar.js — Left_Sidebar behaviour for the standalone Paper Notes Database.
 *
 * Two concerns live here, cleanly separated:
 *
 *   1. A PURE, DOM-free `toggle(state)` function (an involution) that is the
 *      single source of truth for "flip the sidebar state". It is exported for
 *      property testing (Property 12, task 7.5) and takes/returns a boolean.
 *          toggle(true)  === false
 *          toggle(false) === true
 *          toggle(toggle(s)) === s   (two toggles return to the original)
 *
 *   2. DOM glue that wires the `.notes-sidebar-toggle` button to flip the
 *      `.is-collapsed` class on the `#notes-sidebar` element and keep
 *      `aria-expanded` in sync. The state changes ONLY on an explicit user
 *      click of the toggle control — there is deliberately NO resize /
 *      viewport / media-query listener and NO automatic or programmatic
 *      collapse (Requirements 13.4, 13.5). Nearby-paper links are rendered
 *      server-side by _layouts/notes_detail.html (Requirements 13.2, 13.3);
 *      this file does not intercept those links, so normal navigation works.
 *
 * Module convention (shared with assets/notes/search.js so both are importable
 * by the Vitest + fast-check harness): the public API is attached to
 * `module.exports` for CommonJS/ESM-interop consumers (Vitest) AND to
 * `window.NotesSidebar` for the browser. The DOM glue is guarded so it never
 * runs in a non-browser/test context (no real toggle button present).
 */
(function (root, factory) {
  "use strict";

  var api = factory();

  // CommonJS / bundler / Vitest (import interop resolves the default export).
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }

  // Browser global.
  if (typeof window !== "undefined") {
    window.NotesSidebar = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  /**
   * Pure involution over the sidebar's open/closed state.
   *
   * `state` is a boolean where `true` means the sidebar is OPEN (expanded) and
   * `false` means it is CLOSED (collapsed). The function flips the value and is
   * its own inverse, so `toggle(toggle(s)) === s` for every boolean `s`.
   *
   * @param {boolean} state current open state
   * @returns {boolean} the opposite open state
   */
  function toggle(state) {
    return !state;
  }

  /**
   * Reflect an open-state boolean onto a sidebar element and its toggle button.
   * Open  => no `.is-collapsed`, aria-expanded="true".
   * Closed => `.is-collapsed`,   aria-expanded="false".
   *
   * Pure-ish DOM writer used by the click glue; kept separate so the state
   * decision (via `toggle`) stays isolated from the DOM mutation.
   *
   * @param {Element} sidebar the `#notes-sidebar` element
   * @param {Element} button the `.notes-sidebar-toggle` control
   * @param {boolean} open the desired open state
   */
  function applyState(sidebar, button, open) {
    if (sidebar) {
      sidebar.classList.toggle("is-collapsed", !open);
    }
    if (button) {
      button.setAttribute("aria-expanded", open ? "true" : "false");
    }
  }

  /**
   * Wire a single toggle button + sidebar pair. The ONLY thing that changes the
   * sidebar state is a click on the toggle button; we read the current state
   * from the DOM, run it through the pure `toggle`, and write the result back.
   * No other listeners (resize, matchMedia, etc.) are attached, so the state is
   * never changed automatically (Requirement 13.5).
   *
   * @param {Element} sidebar the `#notes-sidebar` element
   * @param {Element} button the `.notes-sidebar-toggle` control
   */
  function wireToggle(sidebar, button) {
    if (!button || !sidebar) {
      return;
    }
    button.addEventListener("click", function () {
      // Current open state is derived from the DOM (is-collapsed => closed).
      var open = !sidebar.classList.contains("is-collapsed");
      applyState(sidebar, button, toggle(open));
    });
  }

  var CONTEXT_KEY = "notesNearbyCtx";

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Read the search/filter context the reader arrived from (sessionStorage).
  function readContext() {
    try {
      var raw = window.sessionStorage.getItem(CONTEXT_KEY);
      if (!raw) return null;
      var ctx = JSON.parse(raw);
      if (!ctx || !Array.isArray(ctx.items)) return null;
      return ctx;
    } catch (e) {
      return null;
    }
  }

  // Fill the "搜索结果" panel from the stored context, highlighting the current
  // paper. Returns true iff the context contains the current paper.
  function renderSearchPanel(doc, sidebar, ctx) {
    var list = doc.getElementById("notes-sidebar-search-list");
    var labelEl = doc.getElementById("notes-sidebar-ctx-label");
    var emptyEl = doc.getElementById("notes-sidebar-ctx-empty");
    if (!list) return false;

    if (!ctx || ctx.items.length === 0) {
      list.innerHTML = "";
      if (labelEl) labelEl.textContent = "";
      if (emptyEl) emptyEl.hidden = false;
      return false;
    }

    var currentUrl = sidebar.getAttribute("data-current-url");
    var containsCurrent = false;
    var html = "";
    for (var i = 0; i < ctx.items.length; i++) {
      var item = ctx.items[i];
      var isCurrent = item.url === currentUrl;
      if (isCurrent) containsCurrent = true;
      html +=
        '<li><a class="notes-sidebar__link' +
        (isCurrent ? " is-current" : "") +
        '" href="' + escapeHtml(item.url) + '">' + escapeHtml(item.title) + "</a></li>";
    }
    list.innerHTML = html;
    if (labelEl) labelEl.textContent = ctx.label || "搜索结果";
    if (emptyEl) emptyEl.hidden = true;
    return containsCurrent;
  }

  // Scroll the current paper to the vertical centre of its list — but only the
  // list's own scroll box, never the page.
  function centerCurrent(panel) {
    if (!panel) return;
    var list = panel.querySelector(".notes-sidebar__list");
    var current = panel.querySelector(".notes-sidebar__link.is-current");
    if (!list || !current) return;
    var listRect = list.getBoundingClientRect();
    var curRect = current.getBoundingClientRect();
    list.scrollTop += (curRect.top - listRect.top) - list.clientHeight / 2 + current.offsetHeight / 2;
  }

  // Wire the 会议+大类 / 搜索结果 mode buttons and pick the initial mode.
  function initContext(doc, sidebar) {
    var modeButtons = sidebar.querySelectorAll(".notes-sidebar__mode");
    if (!modeButtons.length) return;

    var panels = sidebar.querySelectorAll(".notes-sidebar__panel");
    var ctx = readContext();
    var containsCurrent = renderSearchPanel(doc, sidebar, ctx);

    function show(mode) {
      var active = null;
      for (var i = 0; i < panels.length; i++) {
        var isActive = panels[i].getAttribute("data-panel") === mode;
        panels[i].hidden = !isActive;
        if (isActive) active = panels[i];
      }
      for (var j = 0; j < modeButtons.length; j++) {
        modeButtons[j].classList.toggle("is-active", modeButtons[j].getAttribute("data-mode") === mode);
      }
      centerCurrent(active);
    }

    for (var k = 0; k < modeButtons.length; k++) {
      modeButtons[k].addEventListener("click", function () {
        show(this.getAttribute("data-mode"));
      });
    }

    // Navigating via the 会议+大类 list clears any stored search context, so the
    // next paper opens in 会议+大类 (not auto-switched into 搜索结果).
    var confPanel = sidebar.querySelector('.notes-sidebar__panel[data-panel="conf"]');
    if (confPanel) {
      confPanel.addEventListener("click", function (e) {
        var link = e.target.closest ? e.target.closest(".notes-sidebar__link") : null;
        if (!link) return;
        try {
          window.sessionStorage.removeItem(CONTEXT_KEY);
        } catch (e2) {
          /* best-effort */
        }
      });
    }

    // Auto-select the search context when the reader came from a result set
    // that includes this paper; otherwise default to 会议+大类.
    show(containsCurrent ? "search" : "conf");
  }

  /**
   * Find the sidebar/toggle in the document and wire them. Safe to call when no
   * sidebar exists (listing/home/search pages) — it simply does nothing.
   *
   * @param {Document} doc
   */
  function init(doc) {
    if (!doc || typeof doc.querySelector !== "function") {
      return;
    }
    var sidebar = doc.getElementById
      ? doc.getElementById("notes-sidebar")
      : doc.querySelector("#notes-sidebar");
    var button = doc.querySelector(".notes-sidebar-toggle");
    wireToggle(sidebar, button);
    if (sidebar) {
      initContext(doc, sidebar);
    }
  }

  // ---- DOM glue (guarded so it never runs under test / non-browser) ----------
  // Only auto-initialise when there is a real browser document. The property
  // test imports this module in a non-browser (or DOM-less) context and only
  // touches the pure `toggle`, so this block stays dormant there.
  if (typeof document !== "undefined" && typeof window !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () {
        init(document);
      });
    } else {
      init(document);
    }
  }

  return {
    toggle: toggle,
    // Exposed for potential unit tests / reuse; not required by Property 12.
    applyState: applyState,
    wireToggle: wireToggle,
    init: init
  };
});
