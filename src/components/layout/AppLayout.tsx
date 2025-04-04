import { Box, Flex } from "@chakra-ui/react";
import { useRouter } from "next/router";
import { useState, useEffect, type ReactNode } from "react";
import { motion } from "framer-motion";

// Define the Motion Box component
const MotionBox = motion(Box);

// Custom loading animation for route changes
function LoadingAnimation() {
  return (
    <Box
      position="relative"
      width="100px"
      height="100px"
      display="flex"
      alignItems="center"
      justifyContent="center"
    >
      <MotionBox
        initial={{ scale: 0.6, opacity: 1 }}
        animate={{
          scale: [0.6, 1, 0.6],
          opacity: [0.8, 1, 0.8],
        }}
        transition={{
          duration: "2",
          repeat: Infinity,
          ease: "easeInOut",
        }}
      >
        <Box
          as="img"
          src="/tensara_logo_notext.png"
          alt="Tensara Logo"
          width="80px"
          height="80px"
        />
      </MotionBox>
    </Box>
  );
}

export function AppLayout({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [isRouteChanging, setIsRouteChanging] = useState(false);
  const [loadingKey, setLoadingKey] = useState("");

  useEffect(() => {
    const handleRouteChangeStart = (url: string) => {
      setIsRouteChanging(true);
      setLoadingKey(url);
    };

    const handleRouteChangeComplete = () => {
      setIsRouteChanging(false);
    };

    const handleRouteChangeError = () => {
      setIsRouteChanging(false);
    };

    router.events.on("routeChangeStart", handleRouteChangeStart);
    router.events.on("routeChangeComplete", handleRouteChangeComplete);
    router.events.on("routeChangeError", handleRouteChangeError);

    return () => {
      router.events.off("routeChangeStart", handleRouteChangeStart);
      router.events.off("routeChangeComplete", handleRouteChangeComplete);
      router.events.off("routeChangeError", handleRouteChangeError);
    };
  }, [router]);

  return (
    <Flex direction="column" minHeight="100vh" bg="gray.950">
      <Box flex="1" position="relative">
        {isRouteChanging && (
          <Box
            as={motion.div}
            key={loadingKey}
            position="fixed"
            top="0"
            left="0"
            right="0"
            bottom="0"
            display="flex"
            alignItems="center"
            justifyContent="center"
            backgroundColor="rgba(13, 18, 30, 0.3)"
            zIndex="9999"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: "0.3" }}
          >
            <LoadingAnimation />
          </Box>
        )}
        {children}
      </Box>
    </Flex>
  );
}
