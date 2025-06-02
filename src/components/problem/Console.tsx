import { Box, Text, VStack, HStack, Spinner, Badge, Divider } from "@chakra-ui/react";
import { keyframes } from "@emotion/react";
import { SampleStatus, type SampleStatusType, type SampleOutput } from "~/types/submission";

const pulseAnimation = keyframes`
  0% { opacity: 0.6; }
  50% { opacity: 1; }
  100% { opacity: 0.6; }
`;

const StatusBadge = ({ status, isRunning }: { status: SampleStatusType; isRunning: boolean }) => {
  const getStatusProps = () => {
    switch (status) {
      case SampleStatus.IN_QUEUE:
        return { color: '#569cd6', bg: '#1B1B35', text: 'Queued', loading: true };
      case SampleStatus.COMPILING:
        return { color: '#569cd6', bg: '#1B1B35', text: 'Compiling', loading: true };
      case SampleStatus.RUNNING:
        return { color: '#569cd6', bg: '#1B1B35', text: 'Running', loading: true };
      case SampleStatus.PASSED:
        return { color: '#4EC9B0', bg: '#1B352B', text: 'Passed', loading: false };
      case SampleStatus.FAILED:
        return { color: '#FF5D5D', bg: '#351B1B', text: 'Failed', loading: false };
      case SampleStatus.ERROR:
      case SampleStatus.COMPILE_ERROR:
      case SampleStatus.RUNTIME_ERROR:
        return { color: '#FF5D5D', bg: '#351B1B', text: 'Error', loading: false };
      default:
        return { color: '#858585', bg: '#1A1A1A', text: 'Ready', loading: false };
    }
  };
  
  const props = getStatusProps();
  
  return (
    <HStack spacing={2}>
      <Badge
        px={3}
        py={1}
        color={props.color}
        bg={props.bg}
        border="1px solid"
        borderColor={`${props.color}40`}
        borderRadius="md"
        fontWeight="600"
        fontSize="xs"
        animation={props.loading && isRunning ? `${pulseAnimation} 1.5s infinite ease-in-out` : undefined}
      >
        {props.text}
      </Badge>
      {props.loading && isRunning && (
        <Spinner size="xs" color={props.color} speed="0.8s" thickness="2px" />
      )}
    </HStack>
  );
};

const OutputBox = ({ title, content, type = 'default' }: { 
  title?: string; 
  content?: string; 
  type?: 'input' | 'output' | 'expected' | 'error' | 'default';
}) => {
  if (!content) return null;
  
  const getTypeProps = () => {
    switch (type) {
      case 'input':
        return { 
          color: '#569CD6', 
          borderColor: '#2A2A2A',
          label: 'Input'
        };
      case 'expected':
        return { 
          color: '#4FC1FF', 
          borderColor: '#2A2A2A',
          label: 'Expected Output'
        };
      case 'output':
        return { 
          color: '#4EC9B0', 
          borderColor: '#2A2A2A',
          label: 'Your Output'
        };
      case 'error':
        return { 
          color: '#FF5D5D', 
          borderColor: '#2A2A2A',
          label: 'Error Output'
        };
      default:
        return { 
          color: '#858585', 
          borderColor: '#2A2A2A',
          label: title || 'Output'
        };
    }
  };
  
  const props = getTypeProps();
  
  return (
    <Box
      border="1px solid"
      borderColor={props.borderColor}
      borderRadius="md"
      overflow="hidden"
      bg="#111111"
    >
      <Box 
        px={3} 
        py={1}
        bg="#111111"
      >
        <Text fontSize="xs" fontWeight="500" color={props.color}>
          {props.label}
        </Text>
      </Box>
      <Box px={3} py={2} bg="#111111">
        <Text 
          fontFamily="JetBrains Mono, monospace"
          fontSize="sm"
          color="#D4D4D4"
          whiteSpace="pre-wrap"
          wordBreak="break-word"
          lineHeight="1.5"
        >
          {content}
        </Text>
      </Box>
    </Box>
  );
};

type ConsoleProps = {
  output: SampleOutput | null;
  status: SampleStatusType;
  isRunning: boolean;
};

const ResizableConsole = ({ output, status, isRunning }: ConsoleProps) => {
  return (
    <Box
      w="100%"
      h="100%"
      bg="#111111"
      borderRadius="xl"
      border="1px solid"
      borderColor="gray.800"
      overflow="hidden"
    >
      <Box 
        px={4}
        py={3} 
        h="100%" 
        overflowY="auto"
        css={{
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: 'transparent',
          },
          '&::-webkit-scrollbar-thumb': {
            background: '#383838',
            borderRadius: '4px',
          },
          '&::-webkit-scrollbar-thumb:hover': {
            background: '#454545',
          },
        }}
      >
        {!output && status === SampleStatus.IDLE ? (
          <VStack align="center" justify="center" h="100%" spacing={3}>
            <Text color="#858585" fontSize="lg">
              🧪 Test Results
            </Text>
            <Text color="#858585" fontSize="sm" textAlign="center">
              Hit "Run" to test your code with sample inputs
            </Text>
          </VStack>
        ) : (
          <VStack align="stretch" spacing={4}>
            {/* Status */}
            <HStack justify="space-between" align="center">
              <Text color="#D4D4D4" fontSize="md" fontWeight="600">
                Execution Results
              </Text>
              <StatusBadge status={status} isRunning={isRunning} />
            </HStack>
            
            {/* Results Grid */}
            <VStack align="stretch" spacing={3}>
              <OutputBox content={output?.input} type="input" />
              <OutputBox content={output?.output} type="output" />
              
              {output?.stdout && (
                <OutputBox title="Standard Output" content={output.stdout} type="default" />
              )}
              
              {output?.stderr && (
                <OutputBox content={output.stderr} type="error" />
              )}
              
              {output?.message && (
                <OutputBox title="Error Message" content={output.message} type="error" />
              )}
              
              {output?.details && (
                <OutputBox title="Error Details" content={output.details} type="error" />
              )}
            </VStack>
          </VStack>
        )}
      </Box>
    </Box>
  );
};

export default ResizableConsole;
