import { Box, VStack, HStack, Text, Icon, IconButton, Tooltip } from "@chakra-ui/react";
import {
  FiGrid,
  FiBook,
  FiTarget,
  FiList,
  FiCode,
  FiHelpCircle,
  FiLogOut,
  FiLogIn,
  FiChevronLeft,
  FiChevronRight,
} from "react-icons/fi";
import { signOut, signIn, useSession } from "next-auth/react";
import Link from "next/link";
import { useRouter } from "next/router";

interface SidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

const publicMenuItems = [{ icon: FiHelpCircle, label: "Help", href: "/help" }];

const protectedMenuItems = [
  { icon: FiGrid, label: "Dashboard", href: "/dashboard" },
  { icon: FiBook, label: "Problemsets", href: "/problems" },
  { icon: FiTarget, label: "Practice", href: "/practice" },
  { icon: FiList, label: "Catalog", href: "/catalog" },
  { icon: FiBook, label: "Education", href: "/education" },
  { icon: FiCode, label: "API", href: "/api-docs" },
];

export function Sidebar({ isCollapsed, onToggleCollapse }: SidebarProps) {
  const router = useRouter();
  const { data: session } = useSession();

  const handleLogout = async () => {
    await signOut({
      callbackUrl: "/",
      redirect: true,
    });
  };

  const handleLogin = async () => {
    await signIn("github", {
      callbackUrl: "/dashboard",
    });
  };

  return (
    <Box bg="brand.sidebar" h="full" borderRadius="2xl" py={6} px={isCollapsed ? 2 : 4} position="relative">
      <IconButton
        aria-label="Toggle sidebar"
        icon={isCollapsed ? <FiChevronRight /> : <FiChevronLeft />}
        position="absolute"
        right={-3}
        top={6}
        size="sm"
        onClick={onToggleCollapse}
        bg="brand.sidebar"
        _hover={{ bg: "whiteAlpha.200" }}
        borderRadius="full"
      />
      
      <VStack spacing={2} align="stretch" h="full">
        {session &&
          protectedMenuItems.map((item) => (
            <Link key={item.href} href={item.href} passHref>
              <Tooltip label={isCollapsed ? item.label : ""} placement="right" hasArrow>
                <HStack
                  px={isCollapsed ? 2 : 4}
                  py={3}
                  spacing={3}
                  borderRadius="lg"
                  bg={router.pathname === item.href ? "whiteAlpha.200" : "transparent"}
                  color={router.pathname === item.href ? "brand.primary" : "white"}
                  _hover={{ bg: "whiteAlpha.100" }}
                  cursor="pointer"
                  justify={isCollapsed ? "center" : "flex-start"}
                >
                  <Icon as={item.icon} boxSize={5} />
                  {!isCollapsed && (
                    <Text fontSize="sm" fontWeight="medium">
                      {item.label}
                    </Text>
                  )}
                </HStack>
              </Tooltip>
            </Link>
          ))}

        {publicMenuItems.map((item) => (
          <Link key={item.href} href={item.href} passHref>
            <Tooltip label={isCollapsed ? item.label : ""} placement="right" hasArrow>
              <HStack
                px={isCollapsed ? 2 : 4}
                py={3}
                spacing={3}
                borderRadius="lg"
                bg={router.pathname === item.href ? "whiteAlpha.200" : "transparent"}
                color={router.pathname === item.href ? "brand.primary" : "white"}
                _hover={{ bg: "whiteAlpha.100" }}
                cursor="pointer"
                justify={isCollapsed ? "center" : "flex-start"}
              >
                <Icon as={item.icon} boxSize={5} />
                {!isCollapsed && (
                  <Text fontSize="sm" fontWeight="medium">
                    {item.label}
                  </Text>
                )}
              </HStack>
            </Tooltip>
          </Link>
        ))}

        <Box flex={1} />

        <Tooltip label={isCollapsed ? (session ? "Logout" : "Sign In") : ""} placement="right" hasArrow>
          <HStack
            px={isCollapsed ? 2 : 4}
            py={3}
            spacing={3}
            borderRadius="lg"
            color={session ? "red.400" : "brand.primary"}
            _hover={{ bg: "whiteAlpha.100" }}
            cursor="pointer"
            onClick={session ? handleLogout : handleLogin}
            justify={isCollapsed ? "center" : "flex-start"}
          >
            <Icon as={session ? FiLogOut : FiLogIn} boxSize={5} />
            {!isCollapsed && (
              <Text fontSize="sm" fontWeight="medium">
                {session ? "Logout" : "Sign In"}
              </Text>
            )}
          </HStack>
        </Tooltip>
      </VStack>
    </Box>
  );
}
