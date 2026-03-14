(() => {
  const FORM_SELECTOR = ".md-search__form";
  const INPUT_SELECTOR = '.md-search__input[data-md-component="search-query"]';
  const TOGGLE_SELECTOR = 'input[data-md-toggle="search"]';
  const BOUND_ATTR = "data-search-shortcut-bound";
  const HINT_CLASS = "md-search__shortcut";
  const HINT_HIDDEN_CLASS = "md-search__shortcut--hidden";
  const HINT_FADING_CLASS = "md-search__shortcut--fading";
  const SHORTCUT_KEYS = ["/", "f", "s"];
  const SHORTCUT_HINTS = ["/", "F", "S"];
  const HINT_INTERLUDE_SYMBOL = "•";
  const HINT_CYCLE_MS = 3000;
  const HINT_FADE_MS = 220;
  let hintCycleIndex = 0;
  let hintCycleTimer = null;
  let hintFadeTimer = null;

  function isEditableTarget(target) {
    if (!(target instanceof HTMLElement)) return false;
    if (target.isContentEditable) return true;
    return Boolean(
      target.closest(
        "input, textarea, select, [contenteditable=''], [contenteditable='true']"
      )
    );
  }

  function getSearchToggle() {
    return document.querySelector(TOGGLE_SELECTOR);
  }

  function getSearchInput() {
    return document.querySelector(INPUT_SELECTOR);
  }

  function updateHintVisibility(form, hint) {
    const input = form.querySelector(INPUT_SELECTOR);
    const toggle = getSearchToggle();
    const shouldHide =
      !input ||
      document.activeElement === input ||
      input.value.length > 0 ||
      Boolean(toggle && toggle.checked);
    hint.classList.toggle(HINT_HIDDEN_CLASS, shouldHide);
  }

  function getHints() {
    return Array.from(document.querySelectorAll(`.${HINT_CLASS}`));
  }

  function renderHintSymbols() {
    const symbol = SHORTCUT_HINTS[hintCycleIndex];
    getHints().forEach((hint) => {
      hint.textContent = symbol;
    });
  }

  function animateHintTransition() {
    const hints = getHints();
    if (hints.length === 0) return;

    hints.forEach((hint) => {
      hint.classList.add(HINT_FADING_CLASS);
      hint.textContent = HINT_INTERLUDE_SYMBOL;
    });

    if (hintFadeTimer !== null) {
      window.clearTimeout(hintFadeTimer);
    }

    hintFadeTimer = window.setTimeout(() => {
      hintCycleIndex = (hintCycleIndex + 1) % SHORTCUT_HINTS.length;
      renderHintSymbols();
      hints.forEach((hint) => {
        hint.classList.remove(HINT_FADING_CLASS);
      });
    }, HINT_FADE_MS);
  }

  function restartHintCycle() {
    if (hintCycleTimer !== null) {
      window.clearInterval(hintCycleTimer);
    }
    if (hintFadeTimer !== null) {
      window.clearTimeout(hintFadeTimer);
      hintFadeTimer = null;
    }
    renderHintSymbols();
    hintCycleTimer = window.setInterval(() => {
      animateHintTransition();
    }, HINT_CYCLE_MS);
  }

  function isSearchShortcutKey(event) {
    if (event.key.length !== 1) return false;
    return SHORTCUT_KEYS.includes(event.key.toLowerCase());
  }

  function bindSearchForm(form) {
    if (form.getAttribute(BOUND_ATTR) === "true") return;
    form.setAttribute(BOUND_ATTR, "true");

    const hint = document.createElement("kbd");
    hint.className = HINT_CLASS;
    hint.setAttribute("aria-hidden", "true");
    hint.textContent = SHORTCUT_HINTS[hintCycleIndex];
    form.appendChild(hint);

    const refresh = () => updateHintVisibility(form, hint);
    const input = form.querySelector(INPUT_SELECTOR);
    const toggle = getSearchToggle();

    if (input) {
      input.addEventListener("input", refresh);
      input.addEventListener("focus", refresh);
      input.addEventListener("blur", refresh);
    }
    if (toggle) {
      toggle.addEventListener("change", refresh);
    }

    refresh();
  }

  function initSearchShortcut(root = document) {
    root.querySelectorAll(FORM_SELECTOR).forEach(bindSearchForm);
    restartHintCycle();
  }

  function focusSearchFromShortcut(event) {
    if (event.defaultPrevented || !isSearchShortcutKey(event)) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (isEditableTarget(event.target)) return;

    const toggle = getSearchToggle();
    if (toggle && !toggle.checked) {
      toggle.checked = true;
      toggle.dispatchEvent(new Event("change", { bubbles: true }));
    }

    const input = getSearchInput();
    if (!input) return;

    event.preventDefault();
    window.requestAnimationFrame(() => {
      input.focus();
      input.select();
    });
  }

  document.addEventListener("keydown", focusSearchFromShortcut);

  if (
    typeof window.document$ !== "undefined" &&
    typeof window.document$.subscribe === "function"
  ) {
    window.document$.subscribe(() => initSearchShortcut(document));
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () =>
      initSearchShortcut(document)
    );
  } else {
    initSearchShortcut(document);
  }
})();
