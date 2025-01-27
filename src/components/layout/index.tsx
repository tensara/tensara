import { ReactNode } from "react";
import Head from "next/head";
import { GoogleAnalytics } from "@next/third-parties/google";
import { Box, Container, Flex } from "@chakra-ui/react";
import { Header } from "./header";
import { Providers } from "./providers";

import { env } from "~/env"

interface LayoutProps {
  title?: string;
  children: ReactNode;
  showGPU?: boolean;
  maxWidth?: string;
  ogDescription?: string;
}

export function Layout({
  title,
  children,
  showGPU = true,
  maxWidth = "container.xl",
  ogDescription = "Tensara.org - Collaborative platform for optimizing CUDA kernels through crowd-sourced GPU benchmarking",
}: LayoutProps) {
  return (
    <Providers>
      <Head>
        <title>{title ? `${title} | Tensara` : "Tensara.org"}</title>
        <meta name="description" content={ogDescription} />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="icon" href="/favicon.ico" />

        {/* Open Graph */}
        <meta property="og:type" content="website" />
        <meta property="og:site_name" content="Tensara.org" />
        <meta property="og:title" content={title || "Tensara.org"} />
        <meta property="og:description" content={ogDescription} />
        <meta property="og:url" content="https://tensara.org" />
        <meta property="og:image" content="https://tensara.org/og-banner.png" />
        <meta property="og:image:alt" content="Tensara.org platform overview" />

        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:site" content="@tensara_org" />
        <meta name="twitter:creator" content="@tensara_org" />
        <meta name="twitter:title" content={title || "Tensara.org"} />
        <meta name="twitter:description" content={ogDescription} />
        <meta name="twitter:image" content="https://tensara.org/og-banner.png" />

        {/* Security Headers */}
        <meta httpEquiv="Content-Security-Policy" content="default-src 'self' tensara.org" />
        <meta httpEquiv="Permissions-Policy" content="interest-cohort=()" />
      </Head>

      <GoogleAnalytics gaId={env.NEXT_PUBLIC_GA_ID} />

      <Flex direction="column" minH="100vh" bg="gray.900" color="gray.100">
        <Header />

        <Container
          as="main"
          role="main"
          flex="1"
          maxW={maxWidth}
          py={8}
          px={{ base: 4, md: 6, xl: 8 }}
        >
          {children}
        </Container>

        {/* System Status Footer */}
        <Box
          as="footer"
          borderTop="1px solid"
          borderColor="gray.700"
          mt={8}
        >
          <Container maxW={maxWidth} py={4} fontSize="sm">
            <Flex justify="space-between" wrap="wrap">
              <Box>© {new Date().getFullYear()} Tensara.org</Box>
              <Flex gap={4}>
                <a href="https://status.tensara.org" target="_blank" rel="noopener">
                  System Status
                </a>
                <a href="/privacy">Privacy</a>
                <a href="/terms">Terms</a>
              </Flex>
            </Flex>
          </Container>
        </Box>
      </Flex>
    </Providers>
  );
}
