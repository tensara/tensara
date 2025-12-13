import { useEffect, useRef } from "react";

type Options = {
  enabled?: boolean;
  ignoreInputElements?: boolean;
};

export function useHotkey(
  combo: string,
  handler: () => void,
  { enabled = true, ignoreInputElements = true }: Options = {}
) {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    if (!enabled) return;

    const [mod, keyRaw] = combo.toLowerCase().split("+");
    const key = keyRaw;

    const onKeyDown = (e: KeyboardEvent) => {
      if (!enabled) return;

      // 1. Detect Monaco and ALLOW keybinds inside it
      const isMonaco =
        (e.target as HTMLElement)?.closest(".monaco-editor") !== null;

      // 2. Ignore typing inside normal inputs / textareas, BUT NOT MONACO
      if (ignoreInputElements) {
        const tag = (e.target as HTMLElement).tagName.toLowerCase();
        const isNativeInput = tag === "input" || tag === "textarea";

        if (isNativeInput && !isMonaco) {
          return; // block keybinds inside real inputs
        }
      }

      // modifiers
      const metaOk = mod === "meta" ? e.metaKey : true;
      const ctrlOk = mod === "ctrl" ? e.ctrlKey : true;
      const altOk = mod === "alt" ? e.altKey : true;
      const shiftOk = mod === "shift" ? e.shiftKey : true;

      if (metaOk && ctrlOk && altOk && shiftOk) {
        if (e.key.toLowerCase() === key) {
          // Prevent default action and stop propagation so editors
          // (e.g. Monaco) do not also handle this key and insert
          // a newline or other default behavior.
          e.preventDefault();
          e.stopPropagation();
          try {
            e.stopImmediatePropagation();
          } catch {
            /* stopImmediatePropagation may not ben available in some environments */
          }
          handlerRef.current();
        }
      }
    };

    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [combo, enabled, ignoreInputElements]);
}
