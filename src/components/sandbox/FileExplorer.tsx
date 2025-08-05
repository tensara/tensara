import {
  VStack,
  HStack,
  Text,
  IconButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Input,
} from "@chakra-ui/react";
import { FiFile, FiMoreVertical } from "react-icons/fi";
import { useState } from "react";

type SandboxFile = {
  name: string;
  content: string;
};

type FileExplorerProps = {
  files: SandboxFile[];
  active: number;
  onOpen: (index: number) => void;
  onRename: (index: number, name: string) => void;
  onDelete: (index: number) => void;
  onDownload: (index: number) => void;
  readOnly: boolean;
};

export function FileExplorer({
  files,
  active,
  onOpen,
  onRename,
  onDelete,
  onDownload,
  readOnly,
}: FileExplorerProps) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editName, setEditName] = useState("");

  return (
    <VStack align="start" w="100%" spacing={1}>
      {files.map((file: SandboxFile, i: number) => (
        <HStack
          key={file.name + i}
          w="100%"
          px={2}
          py={1}
          bg={i === active ? "#333" : "transparent"}
          borderRadius="md"
          _hover={{ bg: "#2a2a2a" }}
          onClick={() => onOpen(i)}
        >
          <FiFile />
          {editingIndex === i ? (
            <Input
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              size="xs"
              onBlur={() => {
                onRename(i, editName);
                setEditingIndex(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  onRename(i, editName);
                  setEditingIndex(null);
                }
              }}
              autoFocus
            />
          ) : (
            <Text flex={1} fontSize="sm" color="white" isTruncated>
              {file.name}
            </Text>
          )}

          <Menu>
            <MenuButton
              as={IconButton}
              icon={<FiMoreVertical />}
              size="xs"
              variant="ghost"
              aria-label="Options"
              onClick={(e) => e.stopPropagation()}
            />
            <MenuList
              fontSize="sm"
              bg="#2a2a2a"
              color="white"
              border="none"
              boxShadow="md"
            >
              {readOnly == false && (
                <MenuItem
                  bg="#2a2a2a"
                  _hover={{ bg: "#3a3a3a" }}
                  onClick={(e) => {
                    e.stopPropagation();
                    setEditingIndex(i);
                    setEditName(file.name);
                  }}
                >
                  Rename
                </MenuItem>
              )}
              {readOnly == false && (
                <MenuItem
                  bg="#2a2a2a"
                  _hover={{ bg: "#3a3a3a" }}
                  onClick={() => onDelete(i)}
                >
                  Delete
                </MenuItem>
              )}
              <MenuItem
                bg="#2a2a2a"
                _hover={{ bg: "#3a3a3a" }}
                onClick={() => onDownload(i)}
              >
                Download
              </MenuItem>
            </MenuList>
          </Menu>
        </HStack>
      ))}
    </VStack>
  );
}
