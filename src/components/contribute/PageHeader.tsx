import { Box, Heading, Text, useColorModeValue } from "@chakra-ui/react";

interface PageHeaderProps {
  title: string;
  description: string;
}

const PageHeader = ({ title, description }: PageHeaderProps) => {
  const headingColor = useColorModeValue("gray.800", "white");
  const descriptionColor = useColorModeValue("gray.600", "gray.400");

  return (
    <Box mb={10}>
      <Heading
        as="h1"
        size="2xl"
        mb={4}
        color={headingColor}
        fontWeight="bold"
        fontFamily="Space Grotesk, sans-serif"
      >
        {title}
      </Heading>
      <Text fontSize="xl" color={descriptionColor} maxW="2xl">
        {description}
      </Text>
    </Box>
  );
};

export default PageHeader;
