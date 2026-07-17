/*
 * listing.js — progressive "Show more" for the Paper Notes listing pages
 * (conference / category / tag). The layout renders every paper up-front but
 * marks the ones beyond the preview limit with `.is-extra` (hidden via CSS).
 * Each click on `.notes-showmore` reveals the next batch; the button updates
 * its remaining count and removes itself once everything is shown.
 *
 * Pure DOM glue, guarded so it is inert in a non-browser/test context.
 */
(function () {
  "use strict";
  if (typeof document === "undefined") return;

  var BATCH = 10;

  function wire(button) {
    var list = button.parentElement.querySelector(".notes-list");
    if (!list) return;

    button.addEventListener("click", function () {
      var hidden = list.querySelectorAll(".notes-list-row.is-extra");
      for (var i = 0; i < hidden.length && i < BATCH; i++) {
        hidden[i].classList.remove("is-extra");
      }
      var remaining = list.querySelectorAll(".notes-list-row.is-extra").length;
      if (remaining === 0) {
        button.remove();
      } else {
        button.textContent = "Show more (" + remaining + " more)";
      }
    });
  }

  function init() {
    var buttons = document.querySelectorAll(".notes-showmore");
    for (var i = 0; i < buttons.length; i++) wire(buttons[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
