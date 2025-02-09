import {
  Box,
  Button,
  Container,
  Flex,
  HStack,
  Text,
  useColorModeValue,
  Input,
  Icon,
  Image,
} from "@chakra-ui/react";
import { useSession, signIn, signOut } from "next-auth/react";
import Link from "next/link";
import { FiBell, FiTarget } from "react-icons/fi";

export function Header() {
  const { data: session } = useSession();

  return (
    <Box bg="brand.primary" h="full" borderRadius="2xl">
      <Flex
        h="full"
        px={6}
        alignItems="center"
        justifyContent="space-between"
        gap={8}
      >
        <HStack spacing={8}>
          <Link href="/" passHref>
            <HStack spacing={2}>
              <Icon as={FiTarget} boxSize={6} color="white" />
              <Text fontSize="xl" fontWeight="bold" color="white">
                tensara
              </Text>
            </HStack>
          </Link>
        </HStack>

        <Box flex={1} maxW="800px">
          <Input
            placeholder="Find a problem"
            bg="whiteAlpha.300"
            _placeholder={{ color: "whiteAlpha.700" }}
            size="lg"
          />
        </Box>

        <HStack spacing={4}>
          <Icon as={FiBell} boxSize={5} color="white" cursor="pointer" />
          {session?.user?.image && (
            <Image
              src={session.user.image}
              alt="Profile"
              w={8}
              h={8}
              rounded="full"
              border="2px"
              borderColor="white"
            />
          )}
        </HStack>
      </Flex>
    </Box>
  );
}

function NavLink({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) {
  return (
    <Link href={href} passHref>
      <Text
        px={2}
        py={1}
        rounded="md"
        color="gray.300"
        _hover={{
          textDecoration: "none",
          bg: "gray.700",
          color: "white",
        }}
        fontSize="sm"
        fontWeight="medium"
      >
        {children}
      </Text>
    </Link>
  );
}
