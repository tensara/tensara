import { Flex, Box, Text, Link, Button, Divider } from "@chakra-ui/react";

export function Header() {
  return (
    <Box as="header" bg="gray.800" color="white" borderBottom="1px" borderColor="gray.700">
      {/* Top Navigation Bar */}
      <Flex
        px={8}
        py={3}
        justify="space-between"
        align="center"
        fontSize="sm"
        fontWeight="medium"
      >
        {/* Left Side Navigation */}
        <Flex align="center" gap={8}>
          <Text fontSize="xl" fontWeight="bold" color="blue.400">
            tensana
          </Text>
          <Flex gap={6}>
            <Link href="/" _hover={{ textDecoration: "none", color: "blue.400" }}>
              Home
            </Link>
            <Link href="/about" _hover={{ textDecoration: "none", color: "blue.400" }}>
              About
            </Link>
            <Link href="/blog" _hover={{ textDecoration: "none", color: "blue.400" }}>
              Blog
            </Link>
            <Link href="/contact" _hover={{ textDecoration: "none", color: "blue.400" }}>
              Contact
            </Link>
          </Flex>
        </Flex>

        {/* Right Side Action */}
        <Button
          colorScheme="blue"
          size="sm"
          variant="solid"
          _hover={{ transform: "scale(1.02)" }}
        >
          Find a problem
        </Button>
      </Flex>

      {/* Secondary Navigation Bar */}
      <Flex
        px={8}
        py={2}
        bg="gray.850"
        justify="space-between"
        fontSize="sm"
        borderY="1px"
        borderColor="gray.700"
      >
        {/* Dashboard Links */}
        <Flex gap={6}>
          <Link href="/dashboard" _hover={{ color: "blue.400" }}>
            Dashboard
          </Link>
          <Link href="/problemeets" _hover={{ color: "blue.400" }}>
            Problemeets
          </Link>
          <Link href="/practice" _hover={{ color: "blue.400" }}>
            Practice
          </Link>
          <Link href="/catalog" _hover={{ color: "blue.400" }}>
            Catalog
          </Link>
          <Link href="/education" _hover={{ color: "blue.400" }}>
            Education
          </Link>
        </Flex>

        {/* Right Side Links */}
        <Flex gap={6}>
          <Link href="/api" _hover={{ color: "blue.400" }}>
            API
          </Link>
          <Link href="/help" _hover={{ color: "blue.400" }}>
            Help
          </Link>
          <Link href="/logout" _hover={{ color: "blue.400" }}>
            Logout
          </Link>
        </Flex>
      </Flex>
    </Box>
  );
}
