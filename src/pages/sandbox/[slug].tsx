import { useRouter } from "next/router";
import { api } from "~/utils/api";
import { useEffect, useState } from "react";
import SandBox from "./SandboxEditor";

export default function SandboxSlug() {
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

  if (isLoading) return <div>Loading...</div>;
  if (error?.data?.code === "UNAUTHORIZED") return <div>Unauthorized</div>;
  if (!data) return <div>Not found</div>;

  return (
    <SandBox
      files={files}
      setFiles={setFiles}
      main={main}
      setMain={setMain}
      workspaceName={data.name}
      onSave={async () => {
        await update.mutateAsync({ id: data.id, files, main });
      }}
    />
  );
}
