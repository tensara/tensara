import React, { useState } from "react";
import {
  Box,
  HStack,
  VStack,
  Flex,
  Text,
  Icon,
  Button,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  Tooltip,
} from "@chakra-ui/react";

import { FaFire } from "react-icons/fa";
import { CheckIcon, ChevronDownIcon } from "@chakra-ui/icons";

// Define the type for activity data
interface ActivityItem {
  date: string;
  count: number;
}

// Helper function to get ordinal suffix for dates (1st, 2nd, 3rd, etc.)
function getOrdinalSuffix(day: number): string {
  if (day > 3 && day < 21) return "th";
  switch (day % 10) {
    case 1:
      return "st";
    case 2:
      return "nd";
    case 3:
      return "rd";
    default:
      return "th";
  }
}

interface ActivityCalendarProps {
  data: ActivityItem[];
  joinedYear: number;
}

const ActivityCalendar: React.FC<ActivityCalendarProps> = ({
  data,
  joinedYear,
}) => {
  const weeks = 52;
  const days = 7;
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());
  const today = new Date();
  const currentYear = today.getFullYear();

  const dateMap: Record<string, number> = {};
  data.forEach((item) => {
    const itemYear = parseInt(item.date.split("-")[0] ?? "0");
    if (itemYear === selectedYear) {
      dateMap[item.date] = (dateMap[item.date] ?? 0) + item.count;
    }
  });

  type GridCell = { date: string; count: number; bgColor: string };

  const calendarGrid: GridCell[][] = Array(days)
    .fill(null)
    .map(() =>
      Array(weeks)
        .fill(null)
        .map(() => ({ date: "", count: 0, bgColor: "whiteAlpha.100" }))
    );

  const dayNames = ["Mon", "Wed", "Fri", "Sun"];

  const months: string[] = [];

  if (selectedYear === currentYear) {
    for (let i = 0; i < 12; i++) {
      const date = new Date();
      date.setDate(1);
      date.setMonth(today.getMonth() - 11 + i);
      months.push(date.toLocaleString("default", { month: "short" }));
    }
  } else {
    for (let i = 0; i < 12; i++) {
      const date = new Date(selectedYear, i, 1);
      months.push(date.toLocaleString("default", { month: "short" }));
    }
  }

  const isCurrentYear = selectedYear === currentYear;

  for (let w = 0; w < weeks; w++) {
    for (let d = 0; d < days; d++) {
      let date;

      if (isCurrentYear) {
        date = new Date();
        date.setDate(today.getDate() - ((weeks - w - 1) * 7 + (days - d - 1)));
      } else {
        date = new Date(selectedYear, 0, 1);
        const dayOfWeek = date.getDay();

        date.setDate(date.getDate() - (dayOfWeek === 0 ? 6 : dayOfWeek - 1));

        date.setDate(date.getDate() + w * 7 + d);
      }

      const dateStr = date.toISOString().split("T")[0]!;
      const count = dateMap[dateStr] ?? 0;

      let bgColor = "whiteAlpha.100";
      if (count > 0) {
        if (count < 3) bgColor = "green.100";
        else if (count < 6) bgColor = "green.200";
        else if (count < 10) bgColor = "green.400";
        else if (count < 15) bgColor = "green.600";
        else bgColor = "green.700";
      }

      if (calendarGrid[d]) {
        calendarGrid[d]![w] = {
          date: dateStr,
          count,
          bgColor,
        };
      }
    }
  }

  const totalCount = Object.values(dateMap).reduce(
    (sum, count) => sum + count,
    0
  );

  const timeDisplayText =
    selectedYear === currentYear ? "in the last year" : `in ${selectedYear}`;

  const availableYears = [];
  for (let year = currentYear; year >= joinedYear; year--) {
    availableYears.push(year);
  }

  const handleYearChange = (year: number) => {
    setSelectedYear(year);
  };

  return (
    <Box>
      <HStack mb={4} justifyContent="space-between">
        <HStack spacing={3}>
          <Icon as={FaFire} color="blue.300" w={5} h={5} />
          <Text fontSize="sm" color="whiteAlpha.800">
            <Text as="span" fontWeight="bold" fontSize="md" color="white">
              {totalCount}
            </Text>{" "}
            submissions {timeDisplayText}
          </Text>
        </HStack>

        {/* Year dropdown */}
        <Menu>
          <MenuButton
            as={Button}
            size="sm"
            width="100px"
            bg="gray.700"
            color="white"
            borderRadius="lg"
            borderColor="gray.600"
            borderWidth="1px"
            fontWeight="medium"
            _hover={{ borderColor: "blue.300", bg: "gray.650" }}
            _active={{ bg: "gray.650" }}
            rightIcon={<ChevronDownIcon color="blue.300" />}
          >
            {selectedYear}
          </MenuButton>
          <MenuList
            bg="gray.700"
            borderColor="gray.600"
            borderRadius="lg"
            boxShadow="lg"
            py={1}
            minW="100px"
          >
            {availableYears.map((year) => (
              <MenuItem
                key={year}
                value={year}
                onClick={() => handleYearChange(year)}
                bg={selectedYear === year ? "blue.900" : "gray.700"}
                borderRadius="lg"
                color="white"
                _hover={{ bg: "gray.600" }}
                fontSize="sm"
              >
                <Text>{year}</Text>
                {selectedYear === year && (
                  <Icon as={CheckIcon} ml="auto" boxSize={3} />
                )}
              </MenuItem>
            ))}
          </MenuList>
        </Menu>
      </HStack>

      <Flex position="relative">
        {/* Left side with day labels */}
        <Box>
          <Box h="20px"></Box> {/* Empty space to align with graph */}
          <VStack spacing={2} align="flex-start" mt={1}>
            {dayNames.map((day, index) => (
              <Text
                key={day}
                fontSize="xs"
                color="whiteAlpha.600"
                h="10px"
                mt={index > 0 ? "5px" : "0"}
              >
                {day}
              </Text>
            ))}
          </VStack>
        </Box>

        {/* Main calendar area */}
        <Box
          flex="1"
          ml={2}
          key={selectedYear}
          transition="opacity 0.9s ease"
          opacity={1}
          animation="fadeIn 0.9s"
        >
          {/* Month labels */}
          <Flex mb={1} width="100%" ml={4}>
            {months.map((month, i) => (
              <Text
                key={month + i}
                fontSize="xs"
                color="whiteAlpha.700"
                width={`${100 / months.length}%`}
                textAlign="left"
              >
                {month}
              </Text>
            ))}
          </Flex>

          {/* Grid of contribution cells */}
          <Box>
            {calendarGrid.map((row, rowIndex) => (
              <HStack key={rowIndex} spacing={1} mb={1}>
                {row.map((day, colIndex) => (
                  <Tooltip
                    key={`${rowIndex}-${colIndex}`}
                    label={
                      day.date && day.count > 0
                        ? `${day.count} submission${
                            day.count === 1 ? "" : "s"
                          } on ${new Date(day.date).toLocaleDateString(
                            "en-US",
                            {
                              month: "long",
                              day: "numeric",
                            }
                          )}${getOrdinalSuffix(new Date(day.date).getDate())}.`
                        : day.date
                          ? `No submissions on ${new Date(
                              day.date
                            ).toLocaleDateString("en-US", {
                              month: "long",
                              day: "numeric",
                            })}${getOrdinalSuffix(new Date(day.date).getDate())}.`
                          : "No submissions"
                    }
                    placement="top"
                    hasArrow
                    bg="blue.200"
                    color="gray.800"
                    fontSize="xs"
                    px={2}
                    py={1}
                  >
                    <Box w="10px" h="10px" bg={day.bgColor} borderRadius="sm" />
                  </Tooltip>
                ))}
              </HStack>
            ))}

            {/* Less/More spectrum - styled nicely */}
            <Flex justify="flex-end" mt={4} width="100%">
              <Flex
                alignItems="center"
                bg="gray.700"
                py={1}
                px={3}
                borderRadius="md"
              >
                <Text fontSize="xs" color="whiteAlpha.700" mr={2}>
                  Less
                </Text>
                <HStack spacing={1.5}>
                  <Box
                    w="10px"
                    h="10px"
                    borderRadius="sm"
                    bg="whiteAlpha.100"
                  />
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.200" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.400" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.600" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.700" />
                </HStack>
                <Text fontSize="xs" color="whiteAlpha.700" ml={2}>
                  More
                </Text>
              </Flex>
            </Flex>
          </Box>
        </Box>
      </Flex>
    </Box>
  );
};

export default ActivityCalendar;
