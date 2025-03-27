import { Box, type BoxProps } from "@chakra-ui/react";
import { useEffect, useRef } from "react";

interface CodeEditorProps extends Omit<BoxProps, "onChange"> {
  value: string;
  onChange: (value: string) => void;
  language?: string;
  height?: string;
}

export function CodeEditor({
  value,
  onChange,
  height = "300px",
  ...props
}: CodeEditorProps) {
  const editorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Placeholder for actual code editor implementation
    // You can integrate Monaco Editor or CodeMirror here
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.style.width = "100%";
    textarea.style.height = height;
    textarea.style.fontFamily = "monospace";
    textarea.addEventListener("input", (e) => {
      onChange((e.target as HTMLTextAreaElement).value);
    });

    if (editorRef.current) {
      editorRef.current.innerHTML = "";
      editorRef.current.appendChild(textarea);
    }

    return () => {
      textarea.remove();
    };
  }, [height, onChange, value]);

  return <Box ref={editorRef} {...props} />;
}
