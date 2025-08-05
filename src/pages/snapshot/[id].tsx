import { useRouter } from "next/router";
import { useEffect, useState } from "react";
import { Layout } from "~/components/layout";
import SandBox from "~/components/sandbox/SandboxEditor";
import { api } from "~/utils/api";
import WorkspaceAlert from "~/components/sandbox/WorkspaceAlert";

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

  if (isLoading) return <div>Loading...</div>;

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
