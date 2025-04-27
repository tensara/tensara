import {
  Box,
  Flex,
  HStack,
  Text,
  Image,
  Button,
  Icon,
  IconButton,
  useDisclosure,
  VStack,
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
} from "@chakra-ui/react";
import { useSession, signIn, signOut } from "next-auth/react";
import Link from "next/link";
import { useRouter } from "next/router";
import { FiGithub, FiMenu, FiChevronDown } from "react-icons/fi";
import { useState, useEffect } from "react";
import { LayoutGroup, motion } from "framer-motion";
import React from "react";

export function Header() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [isMobile, setIsMobile] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [hoveredPath, setHoveredPath] = useState(router.pathname);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);

    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  const navItems = [
    { label: "Problems", href: "/problems" },
    { label: "Leaderboards", href: "/leaderboard" },
    { label: "Blog", href: "/blog" },
    { label: "Contests", href: "/contests" },
  ];

  const handleSignIn = () => {
    signIn("github", { callbackUrl: router.asPath }).catch(console.error);
  };

  const handleSignOut = () => {
    signOut({ callbackUrl: "/" }).catch(console.error);
  };

  const NavLinks = () => {
    return (
      <LayoutGroup>
        {navItems.map((item) => (
          <Link key={item.href} href={item.href} passHref>
            <Box
              position="relative"
              display="inline-block"
              zIndex={1}
              overflow="hidden"
              borderRadius="0.5rem"
              _hover={{
                cursor: "pointer",
                bg: "rgba(75, 85, 99, 0.5)",
                transition: "all 0.3s ease-in-out",
                transform: "translateY(-1px)",
                boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
              }}
            >
              <Text
                color="white"
                px={3}
                py={2}
                fontSize="0.9rem"
                fontWeight="medium"
                w={isMobile ? "full" : "auto"}
                textAlign={isMobile ? "left" : "center"}
                position="relative"
                zIndex={2}
                cursor="pointer"
              >
                {item.label}
              </Text>
            </Box>
          </Link>
        ))}
      </LayoutGroup>
    );
  };

  const AuthSection = () => {
    const {
      isOpen: menuIsOpen,
      onOpen: menuOnOpen,
      onClose: menuOnClose,
    } = useDisclosure();

    if (!mounted) return null;

    return (
      <>
        {status === "authenticated" ? (
          <Menu isOpen={menuIsOpen} onOpen={menuOnOpen} onClose={menuOnClose}>
            <MenuButton
              as={Box}
              _hover={{
                cursor: "pointer",
                transform: "translateY(-1px)",
                boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
              }}
              transition="all 0.3s ease"
            >
              <HStack spacing={3} py={2} px={7} borderRadius="lg">
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  transition={{
                    duration: 0.3,
                    type: "spring",
                    stiffness: 260,
                    damping: 20,
                  }}
                >
                  <Image
                    src={session.user?.image ?? ""}
                    alt="Profile"
                    w={8}
                    h={8}
                    rounded="full"
                    border="2px solid"
                    borderColor={menuIsOpen ? "brand.primary" : "transparent"}
                    transition="all 0.3s ease"
                  />
                </motion.div>
                <Text color="white" fontSize="sm" fontWeight="medium">
                  {session.user?.username}
                </Text>
                <motion.div
                  style={{
                    display: "inline-flex",
                    transformOrigin: "center center",
                    width: "16px",
                    height: "16px",
                  }}
                  animate={{
                    rotate: menuIsOpen ? 180 : 0,
                  }}
                  transition={{
                    duration: 0.3,
                    type: "spring",
                    stiffness: 200,
                    damping: 25,
                  }}
                >
                  <Icon as={FiChevronDown} color="white" />
                </motion.div>
              </HStack>
            </MenuButton>
            <MenuList
              bg="brand.secondary"
              borderColor="whiteAlpha.200"
              borderRadius="md"
              p={0}
              overflow="hidden"
              minW="180px"
              boxShadow="0 10px 25px -5px rgba(0,0,0,0.3), 0 8px 10px -6px rgba(0,0,0,0.2)"
            >
              <MenuItem
                as={Link}
                href={`/${session.user?.username}`}
                bg="transparent"
                _hover={{
                  bg: "rgba(75, 85, 99, 0.5)",
                  transition: "all 0.3s ease-in-out",
                  boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
                }}
                transition="all 0.15s ease"
                px={4}
                py={2}
                borderRadius="md"
              >
                My Profile
              </MenuItem>
              <MenuItem
                as={Link}
                href="/submissions"
                bg="transparent"
                _hover={{
                  bg: "rgba(75, 85, 99, 0.5)",
                  transition: "all 0.3s ease-in-out",
                  boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
                }}
                transition="all 0.15s ease"
                px={4}
                py={2}
                borderRadius="md"
              >
                Submissions
              </MenuItem>
              <MenuItem
                onClick={handleSignOut}
                bg="transparent"
                _hover={{
                  bg: "rgba(75, 85, 99, 0.5)",
                  transition: "all 0.3s ease-in-out",
                  boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
                }}
                transition="all 0.15s ease"
                px={4}
                py={2}
                borderRadius="md"
              >
                Sign Out
              </MenuItem>
            </MenuList>
          </Menu>
        ) : (
          <Button
            variant="ghost"
            color="white"
            onClick={handleSignIn}
            leftIcon={<Icon as={FiGithub} boxSize={5} />}
            bg="#24292e"
            _hover={{
              bg: "#2f363d",
            }}
          >
            Sign in with GitHub
          </Button>
        )}
      </>
    );
  };

  return (
    <Box bg="" h="full" borderRadius="xl" px={4} pt={2}>
      <Flex h="full" alignItems="center" justifyContent="space-between">
        <HStack>
          <Link href="/" passHref legacyBehavior>
            <HStack as="a">
              <Image
                src="/logo_no_bg.png"
                alt="Tensara Logo"
                w={6}
                h={6}
                mr={1}
                ml={2}
              />
              <Text
                fontSize="xl"
                fontWeight="bold"
                color="white"
                _hover={{ textDecoration: "none" }}
              >
                tensara
              </Text>
            </HStack>
          </Link>

          {/* Desktop Navigation */}
          {!isMobile && (
            <HStack ml={6} spacing={3} pt={2}>
              <NavLinks />
            </HStack>
          )}
        </HStack>

        {/* Mobile Menu Button */}
        {isMobile && (
          <IconButton
            aria-label="Open menu"
            icon={<FiMenu />}
            variant="ghost"
            color="white"
            onClick={onOpen}
          />
        )}

        {/* Desktop Auth Section */}
        {!isMobile && <AuthSection />}

        {/* Mobile Drawer */}
        <Drawer isOpen={isOpen} placement="right" onClose={onClose}>
          <DrawerOverlay />
          <DrawerContent bg="gray.900">
            <DrawerCloseButton color="white" />
            <DrawerHeader borderBottomWidth="1px" color="white">
              Menu
            </DrawerHeader>
            <DrawerBody>
              <VStack align="stretch" spacing={4} mt={4}>
                <NavLinks />
                <Box pt={4} borderTopWidth="1px">
                  <AuthSection />
                </Box>
              </VStack>
            </DrawerBody>
          </DrawerContent>
        </Drawer>
      </Flex>
    </Box>
  );
}

// function NavLink({
//   href,
//   children,
// }: {
//   href: string;
//   children: React.ReactNode;
// }) {
//   return (
//     <Link href={href} passHref>
//       <Text
//         px={2}
//         py={1}
//         rounded="md"
//         color="gray.300"
//         _hover={{
//           textDecoration: "none",
//           bg: "gray.700",
//           color: "white",
//         }}
//         fontSize="sm"
//         fontWeight="medium"
//       >
//         {children}
//       </Text>
//     </Link>
//   );
// }
