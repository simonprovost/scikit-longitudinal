(() => {
  const BOUND_ATTR = "data-api-nav-icon-bound";
  const TOP_LEVEL_SECTION_SELECTOR = "li.md-nav__item--nested";

  const ICONS = {
    "Data Preparation":
      '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-croissant" aria-hidden="true"><path d="M10.2 18H4.774a1.5 1.5 0 0 1-1.352-.97 11 11 0 0 1 .132-6.487M18 10.2V4.774a1.5 1.5 0 0 0-.97-1.352 11 11 0 0 0-6.486.132"/><path d="M18 5a4 3 0 0 1 4 3 2 2 0 0 1-2 2 10 10 0 0 0-5.139 1.42M5 18a3 4 0 0 0 3 4 2 2 0 0 0 2-2 10 10 0 0 1 1.42-5.14"/><path d="M8.709 2.554a10 10 0 0 0-6.155 6.155 1.5 1.5 0 0 0 .676 1.626l9.807 5.42a2 2 0 0 0 2.718-2.718l-5.42-9.807a1.5 1.5 0 0 0-1.626-.676"/></svg>',
    "Data Transformation":
      '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-cooking-pot" aria-hidden="true"><path d="M2 12h20M20 12v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-8M4 8l16-4M8.86 6.78l-.45-1.81a2 2 0 0 1 1.45-2.43l1.94-.48a2 2 0 0 1 2.43 1.46l.45 1.8"/></svg>',
    Preprocessors:
      '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-brain-cog" aria-hidden="true"><path d="m10.852 14.772-.383.923M10.852 9.228l-.383-.923M13.148 14.772l.382.924M13.531 8.305l-.383.923M14.772 10.852l.923-.383M14.772 13.148l.923.383M17.598 6.5A3 3 0 1 0 12 5a3 3 0 0 0-5.63-1.446 3 3 0 0 0-.368 1.571 4 4 0 0 0-2.525 5.771"/><path d="M17.998 5.125a4 4 0 0 1 2.525 5.771"/><path d="M19.505 10.294a4 4 0 0 1-1.5 7.706"/><path d="M4.032 17.483A4 4 0 0 0 11.464 20c.18-.311.892-.311 1.072 0a4 4 0 0 0 7.432-2.516"/><path d="M4.5 10.291A4 4 0 0 0 6 18M6.002 5.125a3 3 0 0 0 .4 1.375M9.228 10.852l-.923-.383M9.228 13.148l-.923.383"/><circle cx="12" cy="12" r="3"/></svg>',
    Estimators:
      '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-brain-circuit" aria-hidden="true"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M9 13a4.5 4.5 0 0 0 3-4M6.003 5.125A3 3 0 0 0 6.401 6.5M3.477 10.896a4 4 0 0 1 .585-.396M6 18a4 4 0 0 1-1.967-.516M12 13h4M12 18h6a2 2 0 0 1 2 2v1M12 8h8M16 8V5a2 2 0 0 1 2-2"/><circle cx="16" cy="13" r=".5"/><circle cx="18" cy="3" r=".5"/><circle cx="20" cy="21" r=".5"/><circle cx="20" cy="8" r=".5"/></svg>',
    Pipeline:
      '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shell" aria-hidden="true"><path d="M14 11a2 2 0 1 1-4 0 4 4 0 0 1 8 0 6 6 0 0 1-12 0 8 8 0 0 1 16 0 10 10 0 1 1-20 0 11.93 11.93 0 0 1 2.42-7.22 2 2 0 1 1 3.16 2.44"/></svg>',
  };

  function addIcon(labelText, container) {
    if (!container || container.getAttribute(BOUND_ATTR) === "true") return;
    const iconMarkup = ICONS[labelText];
    if (!iconMarkup) return;

    container.setAttribute(BOUND_ATTR, "true");
    container.style.display = "inline-flex";
    container.style.alignItems = "center";
    container.style.gap = "0.7rem";

    container.insertAdjacentHTML("afterbegin", iconMarkup);
  }

  function initApiNavIcons(root = document) {
    root.querySelectorAll(TOP_LEVEL_SECTION_SELECTOR).forEach((section) => {
      const sectionLabel = section.querySelector(":scope > label.md-nav__link > .md-ellipsis");
      const sectionText = sectionLabel?.textContent.replace(/\s+/g, " ").trim();

      if (sectionText !== "API") return;

      section
        .querySelectorAll("nav label.md-nav__link > .md-ellipsis")
        .forEach((label) => {
          const text = label.textContent.replace(/\s+/g, " ").trim();
          if (!ICONS[text]) return;

          const container = label.parentElement;
          if (!(container instanceof HTMLElement)) return;

          addIcon(text, container);
        });
    });
  }

  if (
    typeof window.document$ !== "undefined" &&
    typeof window.document$.subscribe === "function"
  ) {
    window.document$.subscribe(() => initApiNavIcons(document));
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initApiNavIcons(document));
  } else {
    initApiNavIcons(document);
  }
})();
