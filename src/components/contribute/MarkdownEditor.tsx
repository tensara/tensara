import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import TurndownService from "turndown";
import showdown from "showdown";
import {
  Box,
  Button,
  Flex,
  Textarea,
  Spinner,
  useColorModeValue,
} from "@chakra-ui/react";

// Dynamically import ReactQuill to avoid SSR issues
const ReactQuill = dynamic(() => import("react-quill"), {
  ssr: false,
  loading: () => <Textarea placeholder="Loading editor..." disabled />,
});

import "react-quill/dist/quill.snow.css";

interface MarkdownEditorProps {
  value: string;
  onChange: (value: string) => void;
  mode: "wysiwyg" | "markdown";
  onModeChange: (mode: "wysiwyg" | "markdown") => void;
}

const MarkdownEditor: React.FC<MarkdownEditorProps> = ({
  value,
  onChange,
  mode,
  onModeChange,
}) => {
  const [isClient, setIsClient] = useState(false);
  const [internalValue, setInternalValue] = useState(value);

  const turndownService = new TurndownService();
  const converter = new showdown.Converter();

  const bgColor = useColorModeValue("white", "gray.800");
  const borderColor = useColorModeValue("gray.200", "gray.600");
  const textColor = useColorModeValue("black", "white");

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    setInternalValue(value);
  }, [value]);

  useEffect(() => {
    if (mode === "markdown") {
      // Switching from Visual (HTML) to Markdown
      const markdown = turndownService.turndown(internalValue);
      onChange(markdown);
    } else if (mode === "wysiwyg") {
      // Switching from Markdown to Visual (HTML)
      const html = converter.makeHtml(internalValue);
      onChange(html);
    }
  }, [mode]);

  const handleQuillChange = (content: string) => {
    setInternalValue(content);
    onChange(content);
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInternalValue(e.target.value);
    onChange(e.target.value);
  };

  return (
    <Box marginTop={4}>
      {mode === "wysiwyg" ? (
        isClient ? (
          <Box
            bg={bgColor}
            borderRadius="lg"
            borderWidth="1px"
            borderColor={borderColor}
            color={textColor}
            boxShadow="sm"
            p={
              0
            } /* Remove padding here as it's handled by Quill's internal padding */
            _hover={{ borderColor: "blue.400" }} /* Add hover effect */
            transition="all 0.2s"
          >
            <ReactQuill
              value={internalValue}
              onChange={handleQuillChange}
              theme="snow"
              modules={{
                toolbar: [
                  [{ header: [1, 2, 3, false] }],
                  ["bold", "italic", "underline", "strike"],
                  [{ list: "ordered" }, { list: "bullet" }],
                  ["link", "code-block"],
                  ["clean"],
                ],
              }}
              /* Custom styling for the toolbar to match the theme */
              /* This targets the .ql-toolbar class */
              className="ql-toolbar-custom"
            />
          </Box>
        ) : (
          <Spinner />
        )
      ) : (
        <Textarea
          value={internalValue}
          onChange={handleTextareaChange}
          rows={10}
          fontFamily="monospace"
          fontSize="sm"
          size="lg"
          bg={bgColor}
          borderColor={borderColor}
          color={textColor}
          p={4} /* Add padding to markdown view for consistency */
          borderRadius="lg" /* Match border radius of WYSIWYG */
          boxShadow="sm" /* Match shadow of WYSIWYG */
          _hover={{ borderColor: "blue.400" }} /* Add hover effect */
          transition="all 0.2s" /* Add transition for hover effect */
        />
      )}
    </Box>
  );
};

export default MarkdownEditor;
