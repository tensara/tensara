import type { ReactNode } from "react";
import Head from "next/head";
import { Box } from "@chakra-ui/react";
import { Header } from "./header";
import { GoogleAnalytics } from "@next/third-parties/google";
import { env } from "~/env";

interface LayoutProps {
  title?: string;
  children: ReactNode;
  ogTitle?: string;
  ogDescription?: string;
  ogImgSubtitle?: string;
  ogImage?: string;
  useDefaultOg?: boolean;
  isCodingMode?: boolean;
  headerToolbar?: ReactNode;
}

export function Layout({
  title,
  children,
  ogTitle = "",
  ogDescription = "A platform for GPU programming challenges. Write efficient GPU kernels and compare your solutions with other developers.",
  ogImgSubtitle = "",
  ogImage,
  useDefaultOg = true,
  isCodingMode = false,
  headerToolbar,
}: LayoutProps) {
  const siteTitle = title ? `${title} | Tensara` : "Tensara";

  // Generate dynamic OG image URL if no custom image is provided and useDefaultOg is true
  const ogImageUrl = ogImage
    ? ogImage.startsWith("http")
      ? ogImage
      : `https://tensara.org${ogImage}`
    : useDefaultOg
      ? `${env.NEXT_PUBLIC_BASE_URL}/api/og?title=${encodeURIComponent(ogTitle)}&subTitle=${encodeURIComponent(ogImgSubtitle)}`
      : undefined;

  return (
    <>
      <Head>
        <title>{siteTitle}</title>
        <meta name="description" content={ogDescription} />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />

        {/* Open Graph tags */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content={siteTitle} />
        <meta property="og:description" content={ogDescription} />
        <meta property="og:url" content="https://tensara.org" />

        {/* Twitter tags */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={siteTitle} />
        <meta name="twitter:description" content={ogDescription} />

        {ogImageUrl && (
          <>
            <meta property="og:image" content={ogImageUrl} />
            <meta name="twitter:image" content={ogImageUrl} />
          </>
        )}
      </Head>

      <GoogleAnalytics gaId={env.NEXT_PUBLIC_GA_ID} />

      <Box h="100vh" bg="brand.dark" display="flex" flexDirection="column">
        <Box p={{ base: 2, md: 2 }}>
          <Header isCodingMode={isCodingMode} toolbar={headerToolbar} />
        </Box>
        <Box
          flex="1"
          borderRadius="xl"
          h="100%"
          px={{ base: 4, md: 2 }}
          py={{ base: isCodingMode ? 0 : 4, md: isCodingMode ? 0 : 2 }}
          overflow="auto"
        >
          {children}
        </Box>
      </Box>
    </>
  );
}
