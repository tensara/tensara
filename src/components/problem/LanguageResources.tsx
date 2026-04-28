import {
  HStack,
  Link as ChakraLink,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  Text,
  Tooltip,
} from "@chakra-ui/react";
import NextLink from "next/link";
import { FaChevronDown } from "react-icons/fa";

import { getLanguageResources } from "~/constants/language";

type LanguageResourcesProps = {
  language: string;
};

export default function LanguageResources({
  language,
}: LanguageResourcesProps) {
  const resources = getLanguageResources(language);

  if (!resources.length) {
    return null;
  }

  return (
    <Menu>
      <Tooltip label="Language resources" hasArrow placement="bottom">
        <MenuButton
          as={ChakraLink}
          _hover={{ color: "white", textDecoration: "none" }}
          _active={{ color: "white" }}
          _focus={{ boxShadow: "none" }}
          fontSize="xs"
          fontWeight="normal"
          h="30px"
          display="inline-flex"
          alignItems="center"
          flexShrink={0}
        >
          <HStack spacing={1} color="gray.300">
            <Text>Resources</Text>
            <FaChevronDown size={8} color="#71717a" />
          </HStack>
        </MenuButton>
      </Tooltip>
      <MenuList bg="brand.secondary" borderColor="gray.800" p={0} minW="180px">
        {resources.map((resource) =>
          resource.isExternal ? (
            <MenuItem
              key={resource.href}
              as={ChakraLink}
              href={resource.href}
              isExternal
              bg="brand.secondary"
              _hover={{ bg: "gray.700", textDecoration: "none" }}
              color="white"
              borderRadius="md"
              fontSize="sm"
            >
              {resource.label}
            </MenuItem>
          ) : (
            <MenuItem
              key={resource.href}
              as={NextLink}
              href={resource.href}
              bg="brand.secondary"
              _hover={{ bg: "gray.700" }}
              color="white"
              borderRadius="md"
              fontSize="sm"
            >
              {resource.label}
            </MenuItem>
          )
        )}
      </MenuList>
    </Menu>
  );
}
