import React from "react";
import {
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Button,
} from "@chakra-ui/react";

interface UserNotFoundAlertProps {
  username: string;
  onReturnHome: () => void;
}

const UserNotFoundAlert: React.FC<UserNotFoundAlertProps> = ({
  username,
  onReturnHome,
}) => {
  return (
    <Alert
      status="error"
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
        User Not Found
      </AlertTitle>
      <AlertDescription maxWidth="sm">
        The user {typeof username === "string" ? username : "User"} doesn&apos;t
        exist or has been removed.
      </AlertDescription>
      <Button
        mt={4}
        colorScheme="white"
        variant="outline"
        onClick={onReturnHome}
      >
        Return to Home
      </Button>
    </Alert>
  );
};

export default UserNotFoundAlert;
