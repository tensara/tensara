import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
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
}

const MarkdownEditor: React.FC<MarkdownEditorProps> = ({ value, onChange }) => {
  const [mode, setMode] = useState<"wysiwyg" | "markdown">("wysiwyg");
  const [isClient, setIsClient] = useState(false);

  const bgColor = useColorModeValue("white", "gray.800");
  const borderColor = useColorModeValue("gray.200", "gray.600");
  const textColor = useColorModeValue("black", "white");

  useEffect(() => {
    setIsClient(true);
  }, []);

  return (
    <Box>
      <Flex mb={2}>
        <Button
          size="sm"
          onClick={() => setMode("wysiwyg")}
          variant={mode === "wysiwyg" ? "solid" : "outline"}
          colorScheme="blue"
        >
          Visual Editor
        </Button>
        <Button
          size="sm"
          ml={2}
          onClick={() => setMode("markdown")}
          variant={mode === "markdown" ? "solid" : "outline"}
          colorScheme="blue"
        >
          Markdown Mode
        </Button>
      </Flex>

      {mode === "wysiwyg" ? (
        isClient ? (
          <Box
            bg={bgColor}
            borderRadius="0.375rem"
            borderWidth="1px"
            borderColor={borderColor}
            color={textColor}
          >
            <ReactQuill
              value={value}
              onChange={onChange}
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
            />
          </Box>
        ) : (
          <Spinner />
        )
      ) : (
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          rows={10}
          fontFamily="monospace"
          size="lg"
          bg={bgColor}
          borderColor={borderColor}
          color={textColor}
        />
      )}
    </Box>
  );
};

export default MarkdownEditor;
