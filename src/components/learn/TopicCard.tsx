import {
  Box,
  Text,
  HStack,
  Badge,
  Icon,
} from "@chakra-ui/react";
import Link from "next/link";
import type { ReactNode } from "react";

interface TopicCardProps {
  title: string;
  description: string;
  href: string;
  icon: ReactNode;
  puzzleCount: number;
  difficulty: string;
  tag?: string;
}

export function TopicCard({
  title,
  description,
  href,
  icon,
  puzzleCount,
  difficulty,
  tag,
}: TopicCardProps) {
  return (
    <Link href={href} passHref legacyBehavior>
      <Box
        as="a"
        display="block"
        bg="brand.secondary"
        border="1px solid"
        borderColor="whiteAlpha.100"
        borderRadius="xl"
        p={6}
        cursor="pointer"
        _hover={{
          borderColor: "brand.primary",
          transform: "translateY(-2px)",
          boxShadow: "0 4px 20px rgba(16, 185, 129, 0.15)",
        }}
        transition="all 0.2s"
        position="relative"
        overflow="hidden"
      >
        {tag && (
          <Badge
            position="absolute"
            top={3}
            right={3}
            colorScheme="green"
            fontSize="2xs"
          >
            {tag}
          </Badge>
        )}

        <Box fontSize="2xl" mb={3}>
          {icon}
        </Box>

        <Text color="white" fontSize="lg" fontWeight="bold" mb={1}>
          {title}
        </Text>

        <Text color="whiteAlpha.600" fontSize="sm" mb={4} noOfLines={2}>
          {description}
        </Text>

        <HStack spacing={3}>
          <Badge variant="subtle" colorScheme="gray" fontSize="2xs">
            {puzzleCount} puzzles
          </Badge>
          <Badge variant="subtle" colorScheme="blue" fontSize="2xs">
            {difficulty}
          </Badge>
        </HStack>
      </Box>
    </Link>
  );
}
