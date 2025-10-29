import { Button, Icon, useDisclosure } from "@chakra-ui/react";
import { FiShare2 } from "react-icons/fi";
import { QuickPostModal } from "./QuickPostModal";

interface ShareSolutionButtonProps {
  submissionId: string;
  problemTitle: string;
  problemSlug: string;
  gflops?: number;
  isVisible: boolean;
}

export function ShareSolutionButton({
  submissionId,
  problemTitle,
  problemSlug,
  gflops,
  isVisible,
}: ShareSolutionButtonProps) {
  const { isOpen, onOpen, onClose } = useDisclosure();

  if (!isVisible) return null;

  return (
    <>
      <Button
        colorScheme="green"
        size="lg"
        leftIcon={<Icon as={FiShare2} />}
        onClick={onOpen}
        bgGradient="linear(to-r, green.500, green.600)"
        _hover={{
          bgGradient: "linear(to-r, green.600, green.700)",
          transform: "translateY(-2px)",
          shadow: "xl",
        }}
        _active={{
          transform: "translateY(0)",
        }}
        transition="all 0.2s"
        fontWeight="semibold"
      >
        Share Your Solution
      </Button>

      <QuickPostModal
        isOpen={isOpen}
        onClose={onClose}
        submissionId={submissionId}
        problemTitle={problemTitle}
        problemSlug={problemSlug}
        gflops={gflops}
      />
    </>
  );
}
