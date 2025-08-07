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

  const router = useRouter();
  const { username, slug } = router.query;

  const { data, isLoading, error } = api.workspace.getBySlug.useQuery(
    username && slug
      ? { username: username as string, slug: slug as string }
      : { username: "", slug: "" }
  );

  const update = api.workspace.update.useMutation();

  const [files, setFiles] = useState<File[]>([]);
  const [main, setMain] = useState("");
  const [hasUserEdited, setHasUserEdited] = useState(false);

  const getStorageKey = () => `workspace_${data?.id}_files`;
  const getMainStorageKey = () => `workspace_${data?.id}_main`;

  useEffect(() => {
    if (data && !hasUserEdited) {
      const storedFiles = localStorage.getItem(getStorageKey());
      const storedMain = localStorage.getItem(getMainStorageKey());

      if (storedFiles && storedMain) {
        setFiles(JSON.parse(storedFiles) as File[]);
        setMain(storedMain);
      } else {
        setFiles(data.files as File[]);
        setMain(data.main ?? "");
      }
    }
  }, [data, hasUserEdited]);

  useEffect(() => {
    if (data && files.length > 0) {
      localStorage.setItem(getStorageKey(), JSON.stringify(files));
    }
  }, [files, data]);

  useEffect(() => {
    if (data && main) {
      localStorage.setItem(getMainStorageKey(), main);
    }
  }, [main, data]);

  const saveToDatabase = async (): Promise<void> => {
    if (!data) return;

    try {
      await update.mutateAsync({ id: data.id, files, main });
      localStorage.removeItem(getStorageKey());
      localStorage.removeItem(getMainStorageKey());
      setHasUserEdited(false);
    } catch (error) {
      console.error("Failed to save to database:", error);
      toast({
        title: "Failed to save workspace.",
        status: "error",
        duration: 3000,
        isClosable: true,
        position: "top-right",
      });
    }
  };

  useEffect(() => {
    const handleBeforeUnload = () => {
      if (hasUserEdited) {
        void saveToDatabase();
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [hasUserEdited, saveToDatabase]);

  const handleFilesChange = (newFiles: File[]) => {
    setHasUserEdited(true);
    setFiles(newFiles);
  };

  const handleMainChange = (newMain: string) => {
    setHasUserEdited(true);
    setMain(newMain);
  };

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
        setFiles={handleFilesChange}
        main={main}
        setMain={handleMainChange}
        onSave={saveToDatabase}
        workspaceName={data.name}
        onDelete={onDelete}
        onRename={onRename}
        readOnly={false}
        onBeforeRun={saveToDatabase}
        onBeforeShare={saveToDatabase}
      />
    </Layout>
  );
}
