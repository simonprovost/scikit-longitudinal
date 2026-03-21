(() => {
  const TRIGGER_SELECTOR = ".expandable-media__trigger";
  const DIALOG_CLASS = "image-lightbox";
  const BOUND_ATTR = "data-image-lightbox-bound";

  function getDialog() {
    let dialog = document.querySelector(`dialog.${DIALOG_CLASS}`);
    if (dialog) return dialog;

    dialog = document.createElement("dialog");
    dialog.className = DIALOG_CLASS;

    const surface = document.createElement("div");
    surface.className = "image-lightbox__surface";

    const image = document.createElement("img");
    image.className = "image-lightbox__image";
    image.alt = "";

    const close = document.createElement("button");
    close.className = "image-lightbox__close";
    close.type = "button";
    close.setAttribute("aria-label", "Close expanded image");
    close.textContent = "x";

    close.addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) {
        dialog.close();
      }
    });

    surface.appendChild(close);
    surface.appendChild(image);
    dialog.appendChild(surface);
    document.body.appendChild(dialog);

    return dialog;
  }

  function bindTrigger(trigger) {
    if (trigger.getAttribute(BOUND_ATTR) === "true") return;
    trigger.setAttribute(BOUND_ATTR, "true");

    trigger.addEventListener("click", (event) => {
      event.preventDefault();

      const image = trigger.querySelector("img");
      const dialog = getDialog();
      const dialogImage = dialog.querySelector(".image-lightbox__image");
      if (!image || !dialogImage) return;

      dialogImage.src = trigger.getAttribute("href") || image.currentSrc || image.src;
      dialogImage.alt = image.alt || "";
      dialog.showModal();
    });
  }

  function initImageLightbox(root = document) {
    root.querySelectorAll(TRIGGER_SELECTOR).forEach(bindTrigger);
  }

  if (
    typeof window.document$ !== "undefined" &&
    typeof window.document$.subscribe === "function"
  ) {
    window.document$.subscribe(() => initImageLightbox(document));
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initImageLightbox(document));
  } else {
    initImageLightbox(document);
  }
})();
