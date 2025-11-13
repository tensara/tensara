import type { SystemStyleObject } from "@chakra-ui/react";

/**
 * Shared markdown content styles for blog posts.
 * Used in both the blog post view and edit preview.
 */
export const markdownContentStyles: SystemStyleObject = {
  "& h1": {
    fontSize: "2xl",
    fontWeight: "800",
    mt: 4,
    mb: 4,
    color: "white",
  },
  "& h2": {
    fontSize: "xl",
    fontWeight: "700",
    mt: 3,
    mb: 3,
    color: "white",
  },
  "& h3": {
    fontSize: "lg",
    fontWeight: "700",
    mt: 2,
    mb: 2,
    color: "gray.100",
  },
  "& p": {
    fontSize: "lg",
    lineHeight: 1.85,
    mb: 4,
    color: "gray.200",
  },
  "& a": {
    color: "green.300",
    textDecoration: "underline",
    _hover: { color: "green.200" },
  },
  "& ul, & ol": {
    pl: 6,
    mb: 4,
    color: "gray.200",
  },
  "& li": {
    mb: 2,
    fontSize: "lg",
  },
  "& blockquote": {
    borderLeftWidth: "3px",
    borderColor: "whiteAlpha.400",
    pl: 4,
    py: 1,
    fontStyle: "italic",
    color: "gray.300",
    my: 4,
  },
  "& code": {
    bg: "whiteAlpha.200",
    px: 1.5,
    py: 0.5,
    borderRadius: "md",
    fontSize: "0.9em",
    fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace",
    color: "green.200",
  },
  "& pre": {
    bg: "gray.800",
    p: 4,
    borderRadius: "lg",
    overflow: "auto",
    mb: 5,
    borderWidth: "1px",
    borderColor: "whiteAlpha.200",
    fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace",
  },
  "& pre code": {
    bg: "transparent",
    p: 0,
    color: "gray.100",
    fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace",
  },
  "& img": {
    borderRadius: "md",
    my: 6,
  },
  "& hr": {
    borderColor: "whiteAlpha.300",
    my: 10,
  },
  "& table": {
    width: "100%",
    mb: 4,
    borderCollapse: "collapse",
  },
  "& th, & td": {
    borderWidth: "1px",
    borderColor: "whiteAlpha.200",
    p: 3,
    textAlign: "left",
  },
  "& th": {
    bg: "whiteAlpha.100",
    fontWeight: "600",
    color: "white",
  },
  "& td": {
    color: "gray.200",
  },
};

