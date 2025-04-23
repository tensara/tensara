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
  FiLogOut,
  FiGithub,
  FiMenu,
  FiChevronDown,
  FiUser,
  FiCode,
} from "react-icons/fi";
import { useState, useEffect } from "react";
import { LayoutGroup, motion, AnimatePresence } from "framer-motion";
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
                bg: "rgba(75, 85, 99, 0.5)",
                transition: "all 0.3s ease-in-out",
                transform: "translateY(-1px)",
                boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
                cursor: "pointer",
              }}
            >
              <Text
                color="white"
                px={3}
                py={2}
                pt={2.5}
                fontSize="0.9rem"
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
    const {
      isOpen: isMenuOpen,
      onOpen: onMenuOpen,
      onClose: onMenuClose,
      onToggle: onMenuToggle,
    } = useDisclosure();
    const menuRef = React.useRef<HTMLDivElement>(null);

    useEffect(() => {
      const handleClickOutside = (event: MouseEvent) => {
        if (
          menuRef.current &&
          !menuRef.current.contains(event.target as Node)
        ) {
          onMenuClose();
        }
      };

      document.addEventListener("mousedown", handleClickOutside);
      return () => {
        document.removeEventListener("mousedown", handleClickOutside);
      };
    }, [menuRef, onMenuClose]);

    if (!mounted) return null;

    return (
      <>
        {status === "authenticated" ? (
          <Box position="relative" ref={menuRef}>
            <HStack
              spacing={3}
              py={2}
              px={7}
              borderRadius="lg"
              onClick={onMenuToggle}
              _hover={{
                cursor: "pointer",
                transform: "translateY(-1px)",
                boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
              }}
              transition="all 0.3s ease"
            >
              <motion.div
                animate={{
                  scale: isMenuOpen ? 1.05 : 1,
                  rotate: isMenuOpen ? 10 : 0,
                }}
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
                  borderColor={isMenuOpen ? "brand.primary" : "transparent"}
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
                  rotate: isMenuOpen ? 180 : 0,
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

            {isMenuOpen && (
              <AnimatePresence>
                <motion.div
                  initial={{ opacity: 0, y: -20, scale: 0.9 }}
                  animate={{
                    opacity: 1,
                    y: 0,
                    scale: 1,
                    transition: {
                      duration: 0.25,
                      ease: [0.16, 1, 0.3, 1],
                    },
                  }}
                  exit={{
                    opacity: 0,
                    y: -10,
                    scale: 0.95,
                    transition: { duration: 0.2 },
                  }}
                  style={{
                    position: "absolute",
                    top: "100%",
                    right: 0,
                    marginTop: "12px",
                    zIndex: 10,
                    width: "180px",
                    transformOrigin: "top right",
                    filter: "drop-shadow(0 10px 15px rgba(0,0,0,0.3))",
                  }}
                >
                  <Box
                    position="absolute"
                    top="-5px"
                    right="20px"
                    width="0"
                    height="0"
                    borderLeft="5px solid transparent"
                    borderRight="5px solid transparent"
                    borderBottom="5px solid"
                    borderBottomColor="brand.secondary"
                  />
                  <VStack
                    bg="brand.secondary"
                    borderRadius="xl"
                    overflow="hidden"
                    spacing={0}
                    border="1px solid"
                    borderColor="whiteAlpha.200"
                    backdropFilter="blur(10px)"
                    style={{
                      boxShadow:
                        "0 10px 25px -5px rgba(0,0,0,0.3), 0 8px 10px -6px rgba(0,0,0,0.2)",
                    }}
                  >
                    <Link
                      href={`/${session.user?.username}`}
                      passHref
                      legacyBehavior
                    >
                      <Box
                        as="a"
                        w="full"
                        px={4}
                        py={3}
                        _hover={{
                          bg: "whiteAlpha.200",
                          transform: "translateX(3px)",
                        }}
                        onClick={onMenuClose}
                        transition="all 0.15s ease"
                        role="group"
                      >
                        <HStack>
                          <motion.div
                            whileHover={{ scale: 1.2 }}
                            transition={{ duration: 0.2 }}
                          >
                            <Icon
                              as={FiUser}
                              color="brand.primary"
                              _groupHover={{ color: "white" }}
                              transition="color 0.2s"
                            />
                          </motion.div>
                          <Text
                            color="white"
                            fontWeight="medium"
                            _groupHover={{ pl: 1 }}
                            transition="padding 0.15s ease"
                          >
                            My Profile
                          </Text>
                        </HStack>
                      </Box>
                    </Link>

                    <Link href={`/submissions`} passHref legacyBehavior>
                      <Box
                        as="a"
                        w="full"
                        px={4}
                        py={3}
                        _hover={{
                          bg: "whiteAlpha.200",
                          transform: "translateX(3px)",
                        }}
                        onClick={onMenuClose}
                        transition="all 0.15s ease"
                        role="group"
                      >
                        <HStack>
                          <motion.div
                            whileHover={{ scale: 1.2 }}
                            transition={{ duration: 0.2 }}
                          >
                            <Icon
                              as={FiCode}
                              color="brand.primary"
                              _groupHover={{ color: "white" }}
                              transition="color 0.2s"
                            />
                          </motion.div>
                          <Text
                            color="white"
                            fontWeight="medium"
                            _groupHover={{ pl: 1 }}
                            transition="padding 0.15s ease"
                          >
                            Submissions
                          </Text>
                        </HStack>
                      </Box>
                    </Link>

                    <Box
                      w="full"
                      px={4}
                      py={3}
                      _hover={{
                        bg: "whiteAlpha.200",
                        transform: "translateX(3px)",
                      }}
                      onClick={() => {
                        onMenuClose();
                        handleSignOut();
                      }}
                      cursor="pointer"
                      transition="all 0.15s ease"
                      role="group"
                    >
                      <HStack>
                        <motion.div
                          whileHover={{ scale: 1.2 }}
                          transition={{ duration: 0.2 }}
                        >
                          <Icon
                            as={FiLogOut}
                            color="brand.primary"
                            _groupHover={{ color: "white" }}
                            transition="color 0.2s"
                          />
                        </motion.div>
                        <Text
                          color="white"
                          fontWeight="medium"
                          _groupHover={{ pl: 1 }}
                          transition="padding 0.15s ease"
                        >
                          Sign Out
                        </Text>
                      </HStack>
                    </Box>
                  </VStack>
                </motion.div>
              </AnimatePresence>
            )}
          </Box>
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
            <HStack ml={8} spacing={3} pt={2}>
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
