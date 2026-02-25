import {
  Box,
  Text,
  Heading,
  Button,
  VStack,
  HStack,
  Spinner,
} from "@chakra-ui/react";
import type { GetServerSideProps } from "next";
import { Layout } from "~/components/layout";
import { api } from "~/utils/api";
import { useRouter } from "next/router";
import { useSession } from "next-auth/react";
import { FaUsers, FaBook } from "react-icons/fa";
import { db } from "~/server/db";

interface JoinPageProps {
  ogGroupName: string | null;
  ogGroupDescription: string | null;
  ogMemberCount: number;
  ogProblemCount: number;
}

export const getServerSideProps: GetServerSideProps<JoinPageProps> = async (context) => {
  const slug = context.params?.slug as string;
  const code = (context.query.code as string) ?? "";

  const group = await db.group.findUnique({
    where: { slug },
    select: {
      name: true,
      description: true,
      inviteCode: true,
      _count: { select: { members: true, problems: true } },
    },
  });

  if (!group || group.inviteCode !== code) {
    return {
      props: {
        ogGroupName: null,
        ogGroupDescription: null,
        ogMemberCount: 0,
        ogProblemCount: 0,
      },
    };
  }

  return {
    props: {
      ogGroupName: group.name,
      ogGroupDescription: group.description,
      ogMemberCount: group._count.members,
      ogProblemCount: group._count.problems,
    },
  };
};

export default function JoinGroupPage({
  ogGroupName,
  ogGroupDescription,
  ogMemberCount,
  ogProblemCount,
}: JoinPageProps) {
  const router = useRouter();
  const { data: session } = useSession();
  const slug = router.query.slug as string;
  const code = router.query.code as string;

  const ogTitle = ogGroupName ? `Join ${ogGroupName}` : "Join Group";
  const ogDesc = ogGroupDescription || (ogGroupName ? `Join ${ogGroupName} on Tensara.` : undefined);
  const ogSub = ogGroupName ? `${ogMemberCount} members · ${ogProblemCount} problems` : "";

  const { data: preview, isLoading, error } = api.groups.getGroupPreview.useQuery(
    { slug, code },
    { enabled: !!slug && !!code && !!session }
  );

  const joinByCode = api.groups.joinByCode.useMutation({
    onSuccess: (data) => {
      void router.push(`/groups/${data.slug}`);
    },
  });

  if (!session) {
    return (
      <Layout title={ogTitle} ogTitle={ogTitle} ogDescription={ogDesc} ogImgSubtitle={ogSub}>
        <Box maxW="7xl" mx="auto" px={4} py={8}>
          <VStack spacing={6} align="center" py={20}>
            <Heading size="lg" color="white">
              Sign in to join a group
            </Heading>
            <Text color="gray.400">
              You need to be signed in to join a group.
            </Text>
          </VStack>
        </Box>
      </Layout>
    );
  }

  if (!code) {
    return (
      <Layout title={ogTitle} ogTitle={ogTitle} ogDescription={ogDesc} ogImgSubtitle={ogSub}>
        <Box maxW="md" mx="auto" px={4} py={16}>
          <VStack spacing={4} align="center">
            <Heading size="lg" color="white">
              Invalid invite link
            </Heading>
            <Text color="gray.400">This link is missing an invite code.</Text>
          </VStack>
        </Box>
      </Layout>
    );
  }

  if (isLoading) {
    return (
      <Layout title={ogTitle} ogTitle={ogTitle} ogDescription={ogDesc} ogImgSubtitle={ogSub}>
        <Box display="flex" justifyContent="center" alignItems="center" h="50vh">
          <Spinner size="xl" />
        </Box>
      </Layout>
    );
  }

  if (error || !preview) {
    return (
      <Layout title={ogTitle} ogTitle={ogTitle} ogDescription={ogDesc} ogImgSubtitle={ogSub}>
        <Box maxW="md" mx="auto" px={4} py={16}>
          <VStack spacing={4} align="center">
            <Heading size="lg" color="white">
              Invalid invite link
            </Heading>
            <Text color="gray.400">
              {error?.message ?? "This invite link is invalid or has expired."}
            </Text>
          </VStack>
        </Box>
      </Layout>
    );
  }

  return (
    <Layout title={ogTitle} ogTitle={ogTitle} ogDescription={ogDesc} ogImgSubtitle={ogSub}>
      <Box maxW="md" mx="auto" px={4} py={16}>
        <VStack spacing={6} align="stretch">
          <Box
            bg="brand.secondary"
            border="1px solid"
            borderColor="whiteAlpha.100"
            borderRadius="xl"
            p={8}
          >
            <VStack spacing={5}>
              <VStack spacing={2}>
                <Text color="gray.400" fontSize="sm" textTransform="uppercase" letterSpacing="wide">
                  You&apos;ve been invited to join
                </Text>
                <Heading size="lg" color="white" textAlign="center">
                  {preview.name}
                </Heading>
                {preview.description && (
                  <Text color="gray.400" fontSize="sm" textAlign="center">
                    {preview.description}
                  </Text>
                )}
              </VStack>

              <HStack spacing={6}>
                <HStack spacing={1}>
                  <FaUsers color="#a0aec0" size={14} />
                  <Text color="gray.400" fontSize="sm">
                    {preview.memberCount} members
                  </Text>
                </HStack>
                <HStack spacing={1}>
                  <FaBook color="#a0aec0" size={14} />
                  <Text color="gray.400" fontSize="sm">
                    {preview.problemCount} problems
                  </Text>
                </HStack>
              </HStack>

              {preview.alreadyMember ? (
                <VStack spacing={3} w="full">
                  <Text color="green.400" fontSize="sm">
                    You&apos;re already a member of this group.
                  </Text>
                  <Button
                    w="full"
                    bg="rgba(34, 197, 94, 0.1)"
                    color="rgb(34, 197, 94)"
                    _hover={{ opacity: 0.9 }}
                    size="lg"
                    onClick={() => void router.push(`/groups/${preview.slug}`)}
                  >
                    Go to Group
                  </Button>
                </VStack>
              ) : (
                <VStack spacing={3} w="full">
                  {joinByCode.error && (
                    <Text color="red.400" fontSize="sm" w="full">
                      {joinByCode.error.message}
                    </Text>
                  )}
                  <Button
                    w="full"
                    bg="rgba(34, 197, 94, 0.1)"
                    color="rgb(34, 197, 94)"
                    _hover={{ opacity: 0.9 }}
                    size="lg"
                    onClick={() => joinByCode.mutate({ slug, code })}
                    isLoading={joinByCode.isPending}
                  >
                    Join Group
                  </Button>
                </VStack>
              )}
            </VStack>
          </Box>
        </VStack>
      </Box>
    </Layout>
  );
}
