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

    const parts = combo.toLowerCase().split("+");
    const modifiers = parts.slice(0, -1);
    const key = parts[parts.length - 1];

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

      // Check that all specified modifiers are pressed
      const requiresMeta = modifiers.includes("meta");
      const requiresCtrl = modifiers.includes("ctrl");
      const requiresAlt = modifiers.includes("alt");
      const requiresShift = modifiers.includes("shift");

      const metaOk = requiresMeta ? e.metaKey : !e.metaKey;
      const ctrlOk = requiresCtrl ? e.ctrlKey : !e.ctrlKey;
      const altOk = requiresAlt ? e.altKey : !e.altKey;
      const shiftOk = requiresShift ? e.shiftKey : !e.shiftKey;

      if (metaOk && ctrlOk && altOk && shiftOk) {
        const pressedKey = e.key.toLowerCase();
        if (
          ["meta", "ctrl", "alt", "shift", "control", "altgraph"].includes(
            pressedKey
          )
        ) {
          return;
        }
        if (pressedKey === key) {
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
