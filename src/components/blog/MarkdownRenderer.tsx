import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import { Fragment } from "react";

interface MarkdownRendererProps {
  content: string;
}

function optimizeCloudinaryUrl(url: string): string {
  // Cloudinary transformation URLs look like:
  // https://res.cloudinary.com/<cloud>/image/upload/<transformations>/<asset>
  // If no transformations are present, insert lightweight defaults.
  const marker = "/image/upload/";
  const idx = url.indexOf(marker);
  if (idx === -1) return url;
  const after = url.slice(idx + marker.length);

  // If there are already transformations (first segment contains commas and no dots),
  // leave it as-is.
  const firstSeg = after.split("/")[0] ?? "";
  const looksLikeTransform = firstSeg.includes(",") && !firstSeg.includes(".");
  if (looksLikeTransform) return url;

  const transform = "f_auto,q_auto,c_limit,w_1600";
  return url.slice(0, idx + marker.length) + transform + "/" + after;
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
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
    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          img({ src, alt, title }) {
            const safeSrc = typeof src === "string" ? src : "";
            const optimized =
              safeSrc.startsWith("https://res.cloudinary.com/")
                ? optimizeCloudinaryUrl(safeSrc)
                : safeSrc;
            return (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={optimized}
                alt={alt ?? ""}
                title={title}
                loading="lazy"
                decoding="async"
              />
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    );
  }

  // Render segments: text as markdown, submissions as embed cards
  return (
    <>
      {segments.map((segment, index) => (
        <Fragment key={index}>
          {/* {segment.type === "text" ? ( */}
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex, rehypeHighlight]}
            components={{
              img({ src, alt, title }) {
                const safeSrc = typeof src === "string" ? src : "";
                const optimized =
                  safeSrc.startsWith("https://res.cloudinary.com/")
                    ? optimizeCloudinaryUrl(safeSrc)
                    : safeSrc;
                return (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={optimized}
                    alt={alt ?? ""}
                    title={title}
                    loading="lazy"
                    decoding="async"
                  />
                );
              },
            }}
          >
            {segment.content}
          </ReactMarkdown>
        </Fragment>
      ))}
    </>
  );
}
