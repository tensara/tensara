import {
  Box,
  HStack,
  Tabs,
  TabList,
  TabPanels,
  TabPanel,
  Tab,
  Text,
  VStack,
} from "@chakra-ui/react";
import { SampleStatus, SampleStatusType, type SampleOutput } from "~/types/submission";
import { FiTerminal } from "react-icons/fi";

type Props = {
  output: SampleOutput | null;
  status: SampleStatusType;
  isRunning: boolean;
  files: { name: string; content: string }[];
};

export default function SandboxConsole({
  output,
  status,
  isRunning,
}: Props) {
  return (
    <Box
      h="100%"
      w="100%"
      borderRadius="xl"
      overflow="hidden"
      border="1px solid #2A2A2A"
      bg="#1b1b35"
      display="flex"
      flexDirection="column"
    >
      <Tabs variant="unstyled" h="100%" display="flex" flexDirection="column">
        <TabList px={4} py={2} borderBottom="1px solid #2A2A2A">
          <Tab color="#858585" _selected={{ color: "#4EC9B0" }}>
            <HStack spacing={2}>
              <FiTerminal />
              <Text>Console</Text>
            </HStack>
          </Tab>
          {/* Add more tabs like PTX, stderr, etc. later */}
        </TabList>

        <TabPanels flex="1" overflowY="auto" px={4} py={3}>
          <TabPanel px={0}>
            {status === SampleStatus.IDLE && (
              <Text color="#858585">Hit "Run" to see console output.</Text>
            )}
            {output?.stdout && (
              <Text
                fontFamily="JetBrains Mono, monospace"
                fontSize="sm"
                color="#D4D4D4"
                whiteSpace="pre-wrap"
                wordBreak="break-word"
              >
                {output.stdout}
              </Text>
            )}
            {output?.stderr && (
              <Text
                fontFamily="JetBrains Mono, monospace"
                fontSize="sm"
                color="#FF5D5D"
                whiteSpace="pre-wrap"
                wordBreak="break-word"
                mt={4}
              >
                {output.stderr}
              </Text>
            )}
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Box>
  );
}
