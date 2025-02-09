import { Box, VStack, HStack, Text, Icon } from "@chakra-ui/react";
import {
  FiGrid,
  FiBook,
  FiTarget,
  FiList,
  FiCode,
  FiHelpCircle,
  FiLogOut,
  FiLogIn,
} from "react-icons/fi";
import { signOut, signIn, useSession } from "next-auth/react";
import Link from "next/link";
import { useRouter } from "next/router";

const publicMenuItems = [{ icon: FiHelpCircle, label: "Help", href: "/help" }];

const protectedMenuItems = [
  { icon: FiGrid, label: "Dashboard", href: "/dashboard" },
  { icon: FiBook, label: "Problemsets", href: "/problems" },
  { icon: FiTarget, label: "Practice", href: "/practice" },
  { icon: FiList, label: "Catalog", href: "/catalog" },
  { icon: FiBook, label: "Education", href: "/education" },
  { icon: FiCode, label: "API", href: "/api-docs" },
];

export function Sidebar() {
  const router = useRouter();
  const { data: session } = useSession();

  const handleLogout = async () => {
    await signOut({
      callbackUrl: "/", // Redirect to home page after logout
      redirect: true,
    });
  };

  const handleLogin = async () => {
    await signIn("github", {
      callbackUrl: "/dashboard",
    });
  };

  return (
    <Box bg="brand.sidebar" h="full" borderRadius="2xl" py={6} px={4}>
      <VStack spacing={2} align="stretch" h="full">
        {session &&
          protectedMenuItems.map((item) => (
            <Link key={item.href} href={item.href} passHref>
              <HStack
                px={4}
                py={3}
                spacing={3}
                borderRadius="lg"
                bg={
                  router.pathname === item.href
                    ? "whiteAlpha.200"
                    : "transparent"
                }
                color={
                  router.pathname === item.href ? "brand.primary" : "white"
                }
                _hover={{ bg: "whiteAlpha.100" }}
                cursor="pointer"
              >
                <Icon as={item.icon} boxSize={5} />
                <Text fontSize="sm" fontWeight="medium">
                  {item.label}
                </Text>
              </HStack>
            </Link>
          ))}

        {publicMenuItems.map((item) => (
          <Link key={item.href} href={item.href} passHref>
            <HStack
              px={4}
              py={3}
              spacing={3}
              borderRadius="lg"
              bg={
                router.pathname === item.href ? "whiteAlpha.200" : "transparent"
              }
              color={router.pathname === item.href ? "brand.primary" : "white"}
              _hover={{ bg: "whiteAlpha.100" }}
              cursor="pointer"
            >
              <Icon as={item.icon} boxSize={5} />
              <Text fontSize="sm" fontWeight="medium">
                {item.label}
              </Text>
            </HStack>
          </Link>
        ))}

        <Box flex={1} />

        <HStack
          px={4}
          py={3}
          spacing={3}
          borderRadius="lg"
          color={session ? "red.400" : "brand.primary"}
          _hover={{ bg: "whiteAlpha.100" }}
          cursor="pointer"
          onClick={session ? handleLogout : handleLogin}
        >
          <Icon as={session ? FiLogOut : FiLogIn} boxSize={5} />
          <Text fontSize="sm" fontWeight="medium">
            {session ? "Logout" : "Sign In"}
          </Text>
        </HStack>
      </VStack>
    </Box>
  );
}
