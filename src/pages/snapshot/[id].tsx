import { useRouter } from "next/router";
import { useEffect, useState } from "react";
import { Layout } from "~/components/layout";
import SandBox from "~/components/sandbox/SandboxEditor";
import { api } from "~/utils/api";
import WorkspaceAlert from "~/components/sandbox/WorkspaceAlert";

export default function SnapshotPage() {
  const router = useRouter();
  const { id } = router.query;

  const { data, isLoading, error } = api.snapshot.getById.useQuery(
    typeof id === "string" ? { id } : { id: "" },
    { enabled: typeof id === "string" }
  );

  const [files, setFiles] = useState([]);
  const [main, setMain] = useState("");

  useEffect(() => {
    if (data) {
      setFiles(data.files);
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
        setFiles={() => {}}
        main={main}
        setMain={() => {}}
        onSave={async () => {}}
        onManualSave={() => {}}
        workspaceName={`Snapshot ${id}`}
        onDelete={() => {}}
        onRename={() => {}}
        readOnly={true}
      />
    </Layout>
  );
}
