import React from "react";
import {
  Box,
  VStack,
  HStack,
  Text,
  Image,
  Flex,
  Icon,
  Link,
} from "@chakra-ui/react";
import { FiLogOut, FiList, FiUser } from "react-icons/fi";
import { type Session } from "next-auth";
interface UserMenuProps {
  session: Session;
  handleSignOut: () => void;
}

const UserMenu = ({ session, handleSignOut }: UserMenuProps) => {
  const bottomMenuItems = [
    {
      label: "Profile",
      icon: FiUser,
      link: `/${session.user?.username}`,
    },
    {
      label: "My Submissions",
      icon: FiList,
      link: "/submissions",
    },
  ];

  return (
    <Box
      bg="gray.900"
      boxShadow="md"
      borderRadius="md"
      p={4}
      w="100%"
      maxW="300px"
    >
      {/* User Profile Section */}
      <Flex align="center" mb={6}>
        <Image
          src={session.user?.image ?? ""}
          alt="Profile"
          w={12}
          h={12}
          rounded="full"
          mr={3}
        />
        <VStack align="start" spacing={0}>
          <Text fontWeight="bold" fontSize="xl">
            {session.user?.username}
          </Text>
        </VStack>
      </Flex>

      {/* Bottom Menu Items */}
      <VStack align="stretch" spacing={3}>
        {bottomMenuItems.map((item, index) => (
          <Link
            href={item.link}
            key={index}
            _hover={{ textDecoration: "none" }}
          >
            <HStack
              py={2}
              _hover={{ bg: "gray.800" }}
              borderRadius="md"
              px={2}
              cursor="pointer"
            >
              <Icon as={item.icon} boxSize={5} color="gray.200" mr={2} />
              <Text fontSize="md" color="gray.200">
                {item.label}
              </Text>
            </HStack>
          </Link>
        ))}

        {/* Sign Out Button */}
        <HStack
          py={2}
          _hover={{ bg: "gray.800" }}
          borderRadius="md"
          px={2}
          cursor="pointer"
          onClick={handleSignOut}
        >
          <Icon as={FiLogOut} boxSize={5} color="gray.200" mr={2} />
          <Text fontSize="md" color="gray.200">
            Sign out
          </Text>
        </HStack>
      </VStack>
    </Box>
  );
};

export default UserMenu;
