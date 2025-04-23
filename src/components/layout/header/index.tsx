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
} from "@chakra-ui/react";
import { useSession, signIn, signOut } from "next-auth/react";
import Link from "next/link";
import { useRouter } from "next/router";
import {
  FiCode,
  FiList,
  FiBookOpen,
  FiLogOut,
  FiGithub,
  FiMenu,
  FiAward,
} from "react-icons/fi";
import { useState, useEffect } from "react";
import {
  AnimatePresence,
  AnimateSharedLayout,
  LayoutGroup,
  motion,
} from "framer-motion";
import { Span } from "next/dist/trace";

export function Header() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [isMobile, setIsMobile] = useState(false);
  const [mounted, setMounted] = useState(false);

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

  const isActivePath = (path: string) => {
    if (path === "/problems") {
      return (
        router.pathname === "/problems" ||
        router.pathname.startsWith("/problems/")
      );
    }
    return router.pathname === path;
  };

  const handleSignIn = () => {
    signIn("github", { callbackUrl: router.asPath }).catch(console.error);
  };

  const handleSignOut = () => {
    signOut({ callbackUrl: "/" }).catch(console.error);
  };

  const NavLinks = () => {
    const pathname = router.pathname || "/";
    const [hoveredPath, setHoveredPath] = useState(pathname);

    console.log("router.pathname", router.pathname);
    console.log("hoveredPath", hoveredPath);
    console.log(
      "navItems",
      navItems.map((item) => item.href)
    );
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
              onMouseEnter={() => setHoveredPath(item.href)}
            >
              {item.href === hoveredPath && (
                <motion.div
                  initial={false}
                  layoutId="navHighlight"
                  style={{
                    position: "absolute",
                    inset: 0,
                    borderRadius: "0.5rem",
                    backgroundColor: "rgba(75, 85, 99, 0.5)",
                  }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{
                    type: "spring",
                    stiffness: 200,
                    damping: 30,
                    mass: 1,
                  }}
                />
              )}

              <Text
                color="white"
                px={3}
                py={2}
                fontSize="1rem"
                fontWeight="medium"
                w={isMobile ? "full" : "auto"}
                textAlign={isMobile ? "left" : "center"}
                position="relative"
                zIndex={2}
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
    if (!mounted) return null;

    return (
      <>
        {status === "authenticated" ? (
          <HStack spacing={4}>
            <Link href={`/${session.user?.username}`} passHref legacyBehavior>
              <HStack
                as="a"
                spacing={3}
                py={2}
                px={3}
                borderRadius="lg"
                _hover={{
                  bg: "whiteAlpha.200",
                  cursor: "pointer",
                }}
                transition="background-color 0.7s"
              >
                <Image
                  src={session.user?.image ?? ""}
                  alt="Profile"
                  w={8}
                  h={8}
                  rounded="full"
                />
                <Text color="white" fontSize="sm">
                  {session.user?.username}
                </Text>
              </HStack>
            </Link>
            <Button
              variant="ghost"
              size="sm"
              color="white"
              onClick={handleSignOut}
              leftIcon={<Icon as={FiLogOut} boxSize={4} />}
              _hover={{
                bg: "whiteAlpha.200",
              }}
              transition="background-color 0.7s"
            >
              Sign out
            </Button>
          </HStack>
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
    <Box bg="transparent" h="full" borderRadius="xl" px={4} py={2}>
      <Flex h="full" alignItems="center" justifyContent="space-between">
        <HStack>
          <Link href="/" passHref legacyBehavior>
            <HStack as="a">
              <Image
                src="/tensara_logo_notext.png"
                alt="Tensara Logo"
                w={12}
                h={12}
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
            <HStack ml={5} spacing={3} pt={2}>
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
