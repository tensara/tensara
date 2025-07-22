import { useRouter } from "next/router";
import { api } from "~/utils/api";
import { useEffect, useState } from "react";
import SandBox from "./SandboxEditor";
import WorkspaceAlert from "~/components/sandbox/WorkspaceAlert";
import { useToast } from "@chakra-ui/react";

export default function SandboxSlug() {
  const toast = useToast();

const handleManualSave = () => {
  update.mutate(
    { id: data.id, files, main },
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
  const slug = router.query.slug as string;

  const { data, isLoading, error } = api.workspace.getBySlug.useQuery({ slug });
  const update = api.workspace.update.useMutation();

  const [files, setFiles] = useState([]);
  const [main, setMain] = useState("");

  useEffect(() => {
    if (data) {
      setFiles(data.files ?? []);
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


  if (isLoading) return <div>Loading...</div>;

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
    <SandBox
      files={files}
      setFiles={setFiles}
      main={main}
      setMain={setMain}
      onSave={() => update.mutate({ id: data.id, files, main })}
      onManualSave={handleManualSave}
      workspaceName={data.name}
    />
  );
}
