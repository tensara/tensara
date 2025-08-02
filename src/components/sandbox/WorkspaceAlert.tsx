import {
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Button,
  Box,
} from "@chakra-ui/react";
import { useRouter } from "next/router";

interface WorkspaceAlertProps {
  status: "error" | "info" | "warning" | "success";
  title: string;
  description: string;
  actionLabel?: string;
}

export default function WorkspaceAlert({
  status,
  title,
  description,
  actionLabel = "Return Home",
}: WorkspaceAlertProps) {
  const router = useRouter();

  return (
    <Box w="100%" display="flex" justifyContent="center" mt={12}>
      <Box maxW="600px" w="80%">
        <Alert
          status={status}
          variant="solid"
          borderRadius="xl"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
          textAlign="center"
          py={6}
        >
          <AlertIcon boxSize="40px" mr={0} />
          <AlertTitle mt={4} mb={1} fontSize="lg">
            {title}
          </AlertTitle>
          <AlertDescription maxWidth="sm">{description}</AlertDescription>
          <Button
            mt={4}
            colorScheme="white"
            variant="outline"
            onClick={() => router.push("/")}
          >
            {actionLabel}
          </Button>
        </Alert>
      </Box>
    </Box>
  );
}
