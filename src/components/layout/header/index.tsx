import { Box, Flex, HStack, Text, Image, Button, Icon } from "@chakra-ui/react";
import { useSession, signIn, signOut } from "next-auth/react";
import Link from "next/link";
import { useRouter } from "next/router";
import { FiCode, FiList, FiBookOpen, FiLogOut, FiGithub } from "react-icons/fi";

export function Header() {
  const { data: session } = useSession();
  const router = useRouter();

  const navItems = [
    { label: "Problems", href: "/problems", icon: FiCode },
    { label: "Submissions", href: "/submissions", icon: FiList },
    { label: "Blog", href: "/blog", icon: FiBookOpen },
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
    signIn("github", { callbackUrl: router.asPath });
  };

  const handleSignOut = () => {
    signOut({ callbackUrl: "/" });
  };

  return (
    <Box bg="brand.navbar" h="full" borderRadius="xl" px={6} py={4}>
      <Flex h="full" alignItems="center" justifyContent="space-between">
        <HStack spacing={8}>
          <Link href="/" passHref legacyBehavior>
            <Text
              as="a"
              fontSize="xl"
              fontWeight="bold"
              color="white"
              _hover={{ textDecoration: "none" }}
            >
              tensara
            </Text>
          </Link>

          <HStack spacing={3}>
            {navItems.map((item) => (
              <Link key={item.href} href={item.href} passHref legacyBehavior>
                <Button
                  as="a"
                  variant="ghost"
                  px={4}
                  py={2}
                  color="white"
                  bg={
                    isActivePath(item.href) ? "whiteAlpha.200" : "transparent"
                  }
                  _hover={{
                    textDecoration: "none",
                    bg: "whiteAlpha.100",
                  }}
                  leftIcon={<Icon as={item.icon} boxSize={5} />}
                >
                  {item.label}
                </Button>
              </Link>
            ))}
          </HStack>
        </HStack>

        {session ? (
          <HStack spacing={4}>
            <HStack spacing={3}>
              <Image
                src={session.user?.image || ""}
                alt="Profile"
                w={8}
                h={8}
                rounded="full"
                border="2px"
                borderColor="white"
              />
              <Text color="white" fontSize="sm">
                {session.user?.name}
              </Text>
            </HStack>
            <Button
              variant="ghost"
              size="sm"
              color="white"
              onClick={handleSignOut}
              leftIcon={<Icon as={FiLogOut} boxSize={4} />}
              _hover={{
                bg: "whiteAlpha.200",
              }}
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
            _hover={{
              bg: "whiteAlpha.200",
            }}
          >
            Sign in with GitHub
          </Button>
        )}
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
