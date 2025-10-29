import { useState, useEffect } from "react";
import { useRouter } from "next/router";
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
  ModalFooter,
  Button,
  VStack,
  HStack,
  Input,
  Textarea,
  Text,
  RadioGroup,
  Radio,
  Stack,
  Progress,
  useToast,
  Spinner,
  Box,
} from "@chakra-ui/react";
import { api } from "~/utils/api";

interface QuickPostModalProps {
  isOpen: boolean;
  onClose: () => void;
  submissionId: string;
  problemTitle: string;
  problemSlug: string;
  gflops?: number;
}

export function QuickPostModal({
  isOpen,
  onClose,
  submissionId,
  problemTitle,
  problemSlug,
  gflops,
}: QuickPostModalProps) {
  const router = useRouter();
  const toast = useToast();

  // State
  const [step, setStep] = useState(1);
  const [title, setTitle] = useState("");
  const [approach, setApproach] = useState("");
  const [challenges, setChallenges] = useState("");
  const [optimizations, setOptimizations] = useState("");
  const [status, setStatus] = useState<"DRAFT" | "PUBLISHED">("DRAFT");

  // Fetch submission details
  const { data: submission, isLoading: isLoadingSubmission } =
    api.submissions.getForBlogPost.useQuery(
      { submissionId },
      {
        enabled: isOpen,
      }
    );

  // Generate default title when submission data loads
  useEffect(() => {
    if (submission && !title && isOpen) {
      const language = submission.language || "Unknown";
      const gflopsText = submission.gflops
        ? `${submission.gflops.toFixed(2)} GFLOPS `
        : "";
      setTitle(`My ${gflopsText}${language} Solution to ${problemTitle}`);
    }
  }, [submission, title, isOpen, problemTitle]);

  // Create post mutation
  const createPost = api.blogpost.createFromSubmission.useMutation({
    onSuccess: (post) => {
      toast({
        title: "Post created!",
        description:
          status === "PUBLISHED"
            ? "Your solution has been shared with the community."
            : "Your draft has been saved.",
        status: "success",
        duration: 5000,
      });
      onClose();
      // Navigate to the new post
      void router.push(`/blog/${post.slug ?? post.id}`);
    },
    onError: (error) => {
      toast({
        title: "Error creating post",
        description: error.message,
        status: "error",
        duration: 5000,
      });
    },
  });

  const handleSubmit = () => {
    createPost.mutate({
      submissionId,
      title: title || undefined,
      approach: approach || undefined,
      challenges: challenges || undefined,
      optimizations: optimizations || undefined,
      status,
    });
  };

  const handleNext = () => {
    if (step < 3) setStep(step + 1);
  };

  const handleBack = () => {
    if (step > 1) setStep(step - 1);
  };

  const handleClose = () => {
    // Reset state
    setStep(1);
    setTitle("");
    setApproach("");
    setChallenges("");
    setOptimizations("");
    setStatus("DRAFT");
    onClose();
  };

  const renderStepContent = () => {
    if (isLoadingSubmission) {
      return (
        <VStack py={8} spacing={4}>
          <Spinner size="xl" />
          <Text>Loading submission details...</Text>
        </VStack>
      );
    }

    switch (step) {
      case 1:
        return (
          <VStack spacing={4} align="stretch">
            <Box>
              <Text fontWeight="semibold" mb={2}>
                Post Title
              </Text>
              <Input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Enter a catchy title for your post"
                size="lg"
              />
            </Box>

            <Box>
              <Text fontWeight="semibold" mb={2}>
                Publish Status
              </Text>
              <RadioGroup
                value={status}
                onChange={(val) => setStatus(val as "DRAFT" | "PUBLISHED")}
              >
                <Stack direction="row" spacing={4}>
                  <Radio value="DRAFT">Save as Draft</Radio>
                  <Radio value="PUBLISHED">Publish Immediately</Radio>
                </Stack>
              </RadioGroup>
              <Text fontSize="sm" color="gray.500" mt={2}>
                {status === "DRAFT"
                  ? "You can publish it later from your profile"
                  : "Your post will be visible to everyone"}
              </Text>
            </Box>
          </VStack>
        );

      case 2:
        return (
          <VStack spacing={4} align="stretch">
            <Box>
              <Text fontWeight="semibold" mb={2}>
                Your Approach
              </Text>
              <Textarea
                value={approach}
                onChange={(e) => setApproach(e.target.value)}
                placeholder="Describe how you approached this problem..."
                rows={4}
              />
            </Box>

            <Box>
              <Text fontWeight="semibold" mb={2}>
                Challenges You Faced
              </Text>
              <Textarea
                value={challenges}
                onChange={(e) => setChallenges(e.target.value)}
                placeholder="What was tricky? What did you learn?..."
                rows={4}
              />
            </Box>

            <Box>
              <Text fontWeight="semibold" mb={2}>
                Key Optimizations
              </Text>
              <Textarea
                value={optimizations}
                onChange={(e) => setOptimizations(e.target.value)}
                placeholder="What made your solution fast? Any clever tricks?..."
                rows={4}
              />
            </Box>
          </VStack>
        );

      case 3:
        return (
          <VStack spacing={4} align="stretch">
            <Text fontWeight="semibold">Preview</Text>
            <Box borderWidth={1} borderRadius="md" p={4} bg="gray.50">
              <Text fontSize="xl" fontWeight="bold" mb={2}>
                {title}
              </Text>
              {submission && (
                <Text fontSize="sm" color="gray.600" mb={4}>
                  Problem: {problemTitle} • Language: {submission.language}
                  {submission.gflops &&
                    ` • ${submission.gflops.toFixed(2)} GFLOPS`}
                </Text>
              )}
              {approach && (
                <Box mb={3}>
                  <Text fontWeight="semibold">Approach:</Text>
                  <Text fontSize="sm">{approach}</Text>
                </Box>
              )}
              {challenges && (
                <Box mb={3}>
                  <Text fontWeight="semibold">Challenges:</Text>
                  <Text fontSize="sm">{challenges}</Text>
                </Box>
              )}
              {optimizations && (
                <Box mb={3}>
                  <Text fontWeight="semibold">Optimizations:</Text>
                  <Text fontSize="sm">{optimizations}</Text>
                </Box>
              )}
              <Text fontSize="sm" fontStyle="italic" color="gray.500">
                Your code and benchmarks will be embedded in the post.
              </Text>
            </Box>
          </VStack>
        );

      default:
        return null;
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} size="2xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>
          Share Your Solution
          <Text fontSize="sm" fontWeight="normal" color="gray.500" mt={1}>
            Step {step} of 3
          </Text>
        </ModalHeader>
        <ModalCloseButton />

        <Progress value={(step / 3) * 100} size="xs" colorScheme="green" />

        <ModalBody py={6}>{renderStepContent()}</ModalBody>

        <ModalFooter>
          <HStack spacing={3}>
            {step > 1 && (
              <Button onClick={handleBack} variant="ghost">
                Back
              </Button>
            )}
            {step < 3 ? (
              <Button colorScheme="green" onClick={handleNext}>
                Next
              </Button>
            ) : (
              <Button
                colorScheme="green"
                onClick={handleSubmit}
                isLoading={createPost.isPending}
                loadingText="Creating..."
              >
                {status === "PUBLISHED" ? "Publish Post" : "Save Draft"}
              </Button>
            )}
          </HStack>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
