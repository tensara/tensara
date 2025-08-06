import { Box, Flex, Heading } from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { motion } from "framer-motion";

// Create motion components
const MotionBox = motion(Box);
const MotionHeading = motion(Heading);

export default function ContestsPage() {
  return (
    <Layout
      title="Contests | Tensara"
      ogTitle="Contests | Tensara"
      ogDescription="Join programming contests on Tensara."
    >
      <Box maxW="7xl" mx="auto" px={4} py={8}>
        <Flex
          justifyContent="space-between"
          alignItems="center"
          mb={6}
          direction={{ base: "column", sm: "row" }}
          gap={{ base: 4, sm: 0 }}
        >
          <Heading size="lg">Contests</Heading>
        </Flex>

        <MotionBox
          display="flex"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
          h="400px"
          bg="brand.secondary"
          borderRadius="xl"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          position="relative"
          overflow="hidden"
          boxShadow="0 10px 30px rgba(0,0,0,0.25)"
          _before={{
            content: '""',
            position: "absolute",
            top: "0",
            left: "0",
            right: "0",
            bottom: "0",
            bg: "brand.secondary",
            zIndex: 0,
          }}
        >
          {/* Animated glowing orbs */}
          <MotionBox
            position="absolute"
            width="150px"
            height="150px"
            borderRadius="full"
            bg="brand.secondary"
            filter="blur(40px)"
            top="30%"
            left="20%"
            animate={{
              x: [0, 30, 0],
              y: [0, -20, 0],
              scale: [1, 1.2, 1],
              opacity: [0.4, 0.6, 0.4],
            }}
            transition={{
              repeat: Infinity,
              duration: 8,
              ease: "easeInOut",
            }}
          />

          <MotionBox
            position="absolute"
            width="200px"
            height="200px"
            borderRadius="full"
            bg="brand.secondary"
            filter="blur(60px)"
            bottom="10%"
            right="15%"
            animate={{
              x: [0, -40, 0],
              y: [0, 30, 0],
              scale: [1, 1.3, 1],
              opacity: [0.3, 0.5, 0.3],
            }}
            transition={{
              repeat: Infinity,
              duration: 10,
              ease: "easeInOut",
              delay: 1,
            }}
          />

          {/* Main content */}
          <MotionHeading
            size="2xl"
            textAlign="center"
            bg="brand.primary"
            bgClip="text"
            fontWeight="bold"
            letterSpacing="wider"
            zIndex={1}
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            COMING SOON
          </MotionHeading>

          <MotionBox
            width="60px"
            height="4px"
            mt={6}
            mb={8}
            bgGradient="linear(to-r, brand.primary, brand.navbar)"
            borderRadius="full"
            initial={{ width: "0px", opacity: 0 }}
            animate={{ width: "60px", opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            zIndex={1}
          />

          <MotionBox
            textAlign="center"
            maxW="500px"
            color="whiteAlpha.800"
            fontSize="lg"
            zIndex={1}
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.7 }}
          >
            We&apos;re preparing exciting GPU programming contests for you.
            Check back soon for competitive challenges and prizes!
          </MotionBox>
        </MotionBox>
      </Box>
    </Layout>
  );
}
