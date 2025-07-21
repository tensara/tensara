import { type NextPage } from "next";
import { useSession } from "next-auth/react";
import { useRouter } from "next/router";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";

import { api } from "~/utils/api";
import {
  Button,
  FormControl,
  FormErrorMessage,
  FormLabel,
  Input,
  Textarea,
  VStack,
  Heading,
  useToast,
  Box,
  Checkbox,
} from "@chakra-ui/react";

const createContestSchema = z.object({
  name: z.string().min(1, "Name is required"),
  description: z.string().min(1, "Description is required"),
  startTime: z.string().min(1, "Start time is required"),
  endTime: z.string().min(1, "End time is required"),
  private: z.boolean(),
});

type CreateContestInput = z.infer<typeof createContestSchema>;

const CreateContestPage: NextPage = () => {
  const router = useRouter();
  const { data: session, status } = useSession();
  const toast = useToast();

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<CreateContestInput>({
    resolver: zodResolver(createContestSchema),
  });

  const createContest = api.contests.create.useMutation({
    onSuccess: () => {
      toast({
        title: "Contest created.",
        description: "The new contest has been created successfully.",
        status: "success",
        duration: 5000,
        isClosable: true,
      });
      void router.push("/contests");
    },
    onError: (error) => {
      toast({
        title: "An error occurred.",
        description: error.message,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    },
  });

  if (status === "loading") {
    return <p>Loading...</p>;
  }

  // The user must be an authenticated admin to create a contest.
  if (status !== "authenticated" || !session?.user?.isAdmin) {
    return <p>Access Denied</p>;
  }

  const onSubmit = (data: CreateContestInput) => {
    createContest.mutate({
      title: data.name,
      description: data.description,
      startTime: new Date(data.startTime),
      endTime: new Date(data.endTime),
      private: data.private,
    });
  };

  return (
    <Box maxW="xl" mx="auto" mt={10}>
      <Heading mb={6}>Create New Contest</Heading>
      <form onSubmit={handleSubmit(onSubmit)}>
        <VStack spacing={4}>
          <FormControl isInvalid={!!errors.name}>
            <FormLabel htmlFor="name">Name</FormLabel>
            <Input id="name" {...register("name")} />
            <FormErrorMessage>{errors.name?.message}</FormErrorMessage>
          </FormControl>
          <FormControl isInvalid={!!errors.description}>
            <FormLabel htmlFor="description">Description</FormLabel>
            <Textarea id="description" {...register("description")} />
            <FormErrorMessage>{errors.description?.message}</FormErrorMessage>
          </FormControl>
          <FormControl isInvalid={!!errors.startTime}>
            <FormLabel htmlFor="startTime">Start Time</FormLabel>
            <Input
              type="datetime-local"
              id="startTime"
              {...register("startTime")}
            />
            <FormErrorMessage>{errors.startTime?.message}</FormErrorMessage>
          </FormControl>
          <FormControl isInvalid={!!errors.endTime}>
            <FormLabel htmlFor="endTime">End Time</FormLabel>
            <Input
              type="datetime-local"
              id="endTime"
              {...register("endTime")}
            />
            <FormErrorMessage>{errors.endTime?.message}</FormErrorMessage>
          </FormControl>
          <FormControl>
            <Checkbox {...register("private")}>Private Contest</Checkbox>
          </FormControl>
          <Button
            type="submit"
            colorScheme="blue"
            isLoading={isSubmitting}
            width="full"
          >
            Create Contest
          </Button>
        </VStack>
      </form>
    </Box>
  );
};

export default CreateContestPage;
