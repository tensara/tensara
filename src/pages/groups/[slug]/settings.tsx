import {
  Box,
  Text,
  Heading,
  Button,
  VStack,
  HStack,
  Spinner,
  Divider,
  useDisclosure,
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
} from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";
import { useRouter } from "next/router";
import Link from "next/link";
import { FiArrowLeft } from "react-icons/fi";
import { FaSignOutAlt, FaTrash } from "react-icons/fa";
import { useRef } from "react";

export default function GroupSettingsPage() {
  const router = useRouter();
  const slug = router.query.slug as string;

  const { data: group, isLoading } = api.groups.getBySlug.useQuery(
    { slug },
    { enabled: !!slug }
  );

  const utils = api.useUtils();

  const deleteGroup = api.groups.deleteGroup.useMutation({
    onSuccess: async () => {
      await utils.groups.getMyGroups.invalidate();
      void router.push("/groups");
    },
  });

  const leaveGroup = api.groups.leaveGroup.useMutation({
    onSuccess: async () => {
      await utils.groups.getMyGroups.invalidate();
      void router.push("/groups");
    },
  });

  const {
    isOpen: isDeleteOpen,
    onOpen: onDeleteOpen,
    onClose: onDeleteClose,
  } = useDisclosure();
  const {
    isOpen: isLeaveOpen,
    onOpen: onLeaveOpen,
    onClose: onLeaveClose,
  } = useDisclosure();
  const cancelRef = useRef<HTMLButtonElement>(null);

  const isOwner = group?.currentUserRole === "OWNER";

  if (isLoading) {
    return (
      <Layout title="Settings">
        <Box display="flex" justifyContent="center" alignItems="center" h="50vh">
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  if (!group) {
    return (
      <Layout title="Settings">
        <Box maxW="7xl" mx="auto" px={4} py={8}>
          <Text color="gray.400">Group not found or you don&apos;t have access.</Text>
        </Box>
      </Layout>
    );
  }

  return (
    <Layout title={`Settings — ${group.name}`}>
      <Box maxW="3xl" mx="auto" px={4} py={8}>
        <VStack align="stretch" spacing={6}>
          <Link href={`/groups/${slug}`} passHref>
            <HStack
              spacing={1}
              color="gray.400"
              _hover={{ color: "white" }}
              cursor="pointer"
              w="fit-content"
            >
              <FiArrowLeft />
              <Text fontSize="sm">Back to {group.name}</Text>
            </HStack>
          </Link>

          <Heading size="lg" color="white">
            Group Settings
          </Heading>

          {/* Group Info */}
          <Box
            bg="brand.secondary"
            borderRadius="xl"
            border="1px solid"
            borderColor="whiteAlpha.100"
            p={6}
          >
            <VStack align="stretch" spacing={3}>
              <Text color="gray.400" fontSize="sm" fontWeight="medium">
                GROUP INFO
              </Text>
              <HStack justify="space-between">
                <Text color="gray.400">Name</Text>
                <Text color="white">{group.name}</Text>
              </HStack>
              <HStack justify="space-between">
                <Text color="gray.400">Slug</Text>
                <Text color="white">{group.slug}</Text>
              </HStack>
              <HStack justify="space-between">
                <Text color="gray.400">Members</Text>
                <Text color="white">{group._count.members}</Text>
              </HStack>
              <HStack justify="space-between">
                <Text color="gray.400">Problems</Text>
                <Text color="white">{group._count.problems}</Text>
              </HStack>
              <HStack justify="space-between">
                <Text color="gray.400">Your Role</Text>
                <Text color="white">{group.currentUserRole}</Text>
              </HStack>
            </VStack>
          </Box>

          <Divider borderColor="whiteAlpha.100" />

          {/* Danger Zone */}
          <Box
            bg="brand.secondary"
            borderRadius="xl"
            border="1px solid"
            borderColor="red.900"
            p={6}
          >
            <VStack align="stretch" spacing={4}>
              <Text color="red.400" fontSize="sm" fontWeight="medium">
                DANGER ZONE
              </Text>

              {!isOwner && (
                <HStack justify="space-between">
                  <VStack align="start" spacing={0}>
                    <Text color="white" fontWeight="medium">
                      Leave Group
                    </Text>
                    <Text color="gray.400" fontSize="sm">
                      You will lose access to this group.
                    </Text>
                  </VStack>
                  <Button
                    size="sm"
                    variant="outline"
                    colorScheme="red"
                    leftIcon={<FaSignOutAlt />}
                    onClick={onLeaveOpen}
                  >
                    Leave
                  </Button>
                </HStack>
              )}

              {isOwner && (
                <HStack justify="space-between">
                  <VStack align="start" spacing={0}>
                    <Text color="white" fontWeight="medium">
                      Delete Group
                    </Text>
                    <Text color="gray.400" fontSize="sm">
                      This action is irreversible. All group data will be lost.
                    </Text>
                  </VStack>
                  <Button
                    size="sm"
                    variant="outline"
                    colorScheme="red"
                    leftIcon={<FaTrash />}
                    onClick={onDeleteOpen}
                  >
                    Delete
                  </Button>
                </HStack>
              )}
            </VStack>
          </Box>
        </VStack>
      </Box>

      {/* Delete Confirmation */}
      <AlertDialog isOpen={isDeleteOpen} leastDestructiveRef={cancelRef} onClose={onDeleteClose} isCentered>
        <AlertDialogOverlay bg="blackAlpha.700">
          <AlertDialogContent bg="brand.secondary" border="1px solid" borderColor="whiteAlpha.100">
            <AlertDialogHeader color="white">Delete Group</AlertDialogHeader>
            <AlertDialogBody color="gray.300">
              Are you sure you want to delete <strong>{group.name}</strong>? This cannot be undone.
            </AlertDialogBody>
            <AlertDialogFooter>
              <Button ref={cancelRef} variant="ghost" color="gray.400" onClick={onDeleteClose}>
                Cancel
              </Button>
              <Button
                colorScheme="red"
                ml={3}
                onClick={() => deleteGroup.mutate({ groupSlug: slug })}
                isLoading={deleteGroup.isPending}
              >
                Delete
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>

      {/* Leave Confirmation */}
      <AlertDialog isOpen={isLeaveOpen} leastDestructiveRef={cancelRef} onClose={onLeaveClose} isCentered>
        <AlertDialogOverlay bg="blackAlpha.700">
          <AlertDialogContent bg="brand.secondary" border="1px solid" borderColor="whiteAlpha.100">
            <AlertDialogHeader color="white">Leave Group</AlertDialogHeader>
            <AlertDialogBody color="gray.300">
              Are you sure you want to leave <strong>{group.name}</strong>?
            </AlertDialogBody>
            <AlertDialogFooter>
              <Button ref={cancelRef} variant="ghost" color="gray.400" onClick={onLeaveClose}>
                Cancel
              </Button>
              <Button
                colorScheme="red"
                ml={3}
                onClick={() => leaveGroup.mutate({ groupSlug: slug })}
                isLoading={leaveGroup.isPending}
              >
                Leave
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </Layout>
  );
}
