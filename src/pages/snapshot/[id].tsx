import { useRouter } from "next/router";
import { useEffect, useState } from "react";
import { Layout } from "~/components/layout";
import SandBox from "~/components/sandbox/SandboxEditor";
import { api } from "~/utils/api";
import WorkspaceAlert from "~/components/sandbox/WorkspaceAlert";
import { Box, Flex, Text, Spinner } from "@chakra-ui/react";
import { keyframes } from "@emotion/react";

const pulseAnimation = keyframes`
  0% { opacity: 0.6; }
  50% { opacity: 0.8; }
  100% { opacity: 0.6; }
`;

export default function SnapshotPage() {
  const router = useRouter();
  const { id } = router.query;
  const noop = () => {
    // intentionally empty
  };

  const noopAsync = async () => {
    // intentionally empty
  };

  const { data, isLoading, error } = api.snapshot.getById.useQuery(
    typeof id === "string" ? { id } : { id: "" },
    { enabled: typeof id === "string" }
  );

  type File = {
    name: string;
    content: string;
  };

  const [files, setFiles] = useState<File[]>([]);
  const [main, setMain] = useState("");

  useEffect(() => {
    if (data) {
      setFiles(data.files ? (data.files as File[]) : []);
      setMain(data.main);
    }
  }, [data]);

  if (isLoading)
    return (
      <Box
        w="100%"
        h="100%"
        bg="brand.secondary"
        borderRadius="xl"
        overflow="hidden"
        position="relative"
      >
        <Flex
          position="absolute"
          w="100%"
          h="100%"
          zIndex="1"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
        >
          <Box
            position="absolute"
            top="0"
            left="0"
            right="0"
            bottom="0"
            bgGradient="linear(to-b, brand.dark, #191919)"
            animation={`${pulseAnimation} 2s infinite ease-in-out`}
          />

          <Spinner
            size="xl"
            thickness="3px"
            speed="0.8s"
            color="brand.navbar"
            zIndex="2"
            mb="3"
          />

          <Text color="gray.300" fontFamily="mono" fontSize="sm" zIndex="2">
            Loading...
          </Text>
        </Flex>
      </Box>
    );

  if (!data || error) {
    return (
      <WorkspaceAlert
        status="error"
        title="Snapshot Not Found"
        description="The requested snapshot doesn't exist or has been deleted."
      />
    );
  }

  return (
    <Layout title="Snapshot Viewer">
      <SandBox
        files={files}
        setFiles={noop}
        main={main}
        setMain={noop}
        onSave={noopAsync}
        onManualSave={noop}
        workspaceName={`${data.user.name}'s Snapshot`}
        onDelete={noop}
        onRename={noop}
        readOnly={true}
      />
    </Layout>
  );
}
