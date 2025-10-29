import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { SubmissionEmbedCard } from "./SubmissionEmbedCard";
import { Fragment } from "react";

interface SubmissionEmbedRendererProps {
  content: string;
}

/**
 * Parses blog post content and renders submission embeds
 *
 * This component searches for {{submission:id}} markers in markdown content
 * and replaces them with interactive SubmissionEmbedCard components.
 *
 * Example markers:
 * - {{submission:abc123}}
 * - {{submission:def456}}
 *
 * The regex pattern matches:
 * - Opening: {{submission:
 * - Submission ID: alphanumeric, underscores, hyphens
 * - Closing: }}
 */
export function SubmissionEmbedRenderer({
  content,
}: SubmissionEmbedRendererProps) {
  // Regex to match {{submission:submissionId}} patterns
  // Captures the submission ID in group 1
  const submissionMarkerRegex = /\{\{submission:([a-zA-Z0-9_-]+)\}\}/g;

  // Split content into segments of text and submission markers
  const segments: Array<{ type: "text" | "submission"; content: string }> = [];
  let lastIndex = 0;
  let match;

  // Find all submission markers and split content accordingly
  while ((match = submissionMarkerRegex.exec(content)) !== null) {
    // Add text segment before this marker (if any)
    if (match.index > lastIndex) {
      segments.push({
        type: "text",
        content: content.slice(lastIndex, match.index),
      });
    }

    // Add submission marker segment
    segments.push({
      type: "submission",
      content: match[1]!, // The captured submission ID
    });

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text after the last marker (if any)
  if (lastIndex < content.length) {
    segments.push({
      type: "text",
      content: content.slice(lastIndex),
    });
  }

  // If no markers found, render as regular markdown
  if (segments.length === 0) {
    return <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>;
  }

  // Render segments: text as markdown, submissions as embed cards
  return (
    <>
      {segments.map((segment, index) => (
        <Fragment key={index}>
          {segment.type === "text" ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {segment.content}
            </ReactMarkdown>
          ) : (
            <SubmissionEmbedCard submissionId={segment.content} />
          )}
        </Fragment>
      ))}
    </>
  );
}
