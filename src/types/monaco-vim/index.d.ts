declare module "monaco-vim" {
  import type { editor } from "monaco-editor";

  export interface VimModeAdapter {
    dispose(): void;
    getOption?: (key: string) => unknown;
  }

  export function initVimMode(
    editorInstance: editor.IStandaloneCodeEditor,
    statusbar?: HTMLElement | null
  ): VimModeAdapter;
}
