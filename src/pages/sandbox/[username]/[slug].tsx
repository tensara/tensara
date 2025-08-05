import { useRouter } from "next/router";
import { api } from "~/utils/api";
import { keyframes } from "@emotion/react";
import { useEffect, useState } from "react";
import SandBox from "../../../components/sandbox/SandboxEditor";
import WorkspaceAlert from "~/components/sandbox/WorkspaceAlert";
import { useToast } from "@chakra-ui/react";
import { Layout } from "~/components/layout";
import { Box, Flex, Text, Spinner } from "@chakra-ui/react";

const pulseAnimation = keyframes`
  0% { opacity: 0.6; }
  50% { opacity: 0.8; }
  100% { opacity: 0.6; }
`;

export default function SandboxSlug() {
  const toast = useToast();
  type File = {
    name: string;
    content: string;
  };

  const deleteMutation = api.workspace.delete.useMutation({
    onSuccess: () => {
      void router.push("/sandbox");
    },
  });

  const onDelete = () => {
    if (confirm("Are you sure you want to delete this workspace?")) {
      deleteMutation.mutate({ id: data?.id ?? "" });
    }
  };

  const rename = api.workspace.rename.useMutation();
  const utils = api.useUtils();

  const onRename = async (newName: string) => {
    if (!data) return;
    await rename.mutateAsync({ id: data.id, name: newName });
    await utils.workspace.getBySlug.invalidate(); // refresh data
  };

  const handleManualSave = () => {
    update.mutate(
      { id: data?.id ?? "", files, main },
      {
        onSuccess: () => {
          toast({
            title: "Workspace saved.",
            status: "success",
            duration: 3000,
            isClosable: true,
            position: "top-right",
          });
        },
        onError: () => {
          toast({
            title: "Failed to save workspace.",
            status: "error",
            duration: 3000,
            isClosable: true,
            position: "top-right",
          });
        },
      }
    );
  };

  const router = useRouter();
  // const username = router.query.username as string;

  // const slug = router.query.slug as string;
  const { username, slug } = router.query;

  // const { data, isLoading, error } = api.workspace.getBySlug.useQuery({ slug });
  const { data, isLoading, error } = api.workspace.getBySlug.useQuery(
    username && slug
      ? { username: username as string, slug: slug as string }
      : { username: "", slug: "" }
  );

  const update = api.workspace.update.useMutation();

  const [files, setFiles] = useState<File[]>([]);
  const [main, setMain] = useState("");

  useEffect(() => {
    if (data) {
      // setFiles(data.files ?? []);
      setFiles(data.files as File[]);

      setMain(data.main ?? "");
    }
  }, [data]);

  useEffect(() => {
    if (!data) return;

    const timeout = setTimeout(() => {
      update.mutate({ id: data.id, files, main });
    }, 2000); // 2s debounce

    return () => clearTimeout(timeout);
  }, [files, main]);

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

  if (error?.data?.code === "UNAUTHORIZED") {
    return (
      <WorkspaceAlert
        status="error"
        title="Unauthorized"
        description="You do not have permission to access this workspace."
      />
    );
  }

  if (!data) {
    return (
      <WorkspaceAlert
        status="error"
        title="Workspace Not Found"
        description="The requested workspace doesn't exist or has been deleted."
      />
    );
  }

  return (
    <Layout title="Sandbox Editor">
      <SandBox
        files={files}
        setFiles={setFiles}
        main={main}
        setMain={setMain}
        onSave={async () => {
          await update.mutateAsync({ id: data.id, files, main });
        }}
        onManualSave={handleManualSave}
        workspaceName={data.name}
        onDelete={onDelete}
        onRename={onRename}
        readOnly={false}
      />
    </Layout>
  );
}
