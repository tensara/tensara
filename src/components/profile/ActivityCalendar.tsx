import React, { useState, useMemo } from "react";
import {
  Box,
  HStack,
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

import { FaFire, FaCheck, FaChevronDown } from "react-icons/fa";

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
  const { currentYear } = useMemo(() => {
    const today = new Date();
    return { today, currentYear: today.getFullYear() };
  }, []);

  const [selectedYear, setSelectedYear] = useState(currentYear);

  const calendarData = useMemo(() => {
    const weeks = 52;
    const days = 7;

    const dateMap: Record<string, number> = {};
    data.forEach((item) => {
      const itemYear = parseInt(item.date.split("-")[0] ?? "0");
      if (itemYear === selectedYear) {
        dateMap[item.date] = (dateMap[item.date] ?? 0) + item.count;
      }
    });

    // Day labels - show all days for better alignment
    const dayNames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

    // Calculate month positions more accurately
    const monthLabels: { month: string; position: number }[] = [];

    for (let month = 0; month < 12; month++) {
      const firstDay = new Date(selectedYear, month, 1);
      // Space months evenly across the 52 weeks
      const position = Math.floor((month * 52) / 12);

      monthLabels.push({
        month: firstDay.toLocaleString("default", { month: "short" }),
        position,
      });
    }

    type GridCell = { date: string; count: number; bgColor: string };

    // Initialize the grid with empty cells
    const calendarGrid: GridCell[][] = Array.from({ length: days }, () =>
      Array.from({ length: weeks }, () => ({
        date: "",
        count: 0,
        bgColor: "whiteAlpha.100",
      }))
    );

    // Fill the grid with proper dates and data
    const isCurrentYear = selectedYear === currentYear;

    // Create date for January 1st of the selected year
    const jan1 = new Date(selectedYear, 0, 1);

    // Find the first Monday of the year or the last Monday of previous year
    const firstMonday = new Date(jan1);
    // Get days to subtract to reach the previous Monday
    // Sunday is 0, so if it's 0, we need 6 days back to previous Monday
    // If it's Monday (1), we need 0 days back
    const daysToSubtract = (jan1.getDay() + 6) % 7;
    firstMonday.setDate(jan1.getDate() - daysToSubtract);

    // Fill grid with dates
    for (let w = 0; w < weeks; w++) {
      for (let d = 0; d < days; d++) {
        const cellDate = new Date(firstMonday);
        cellDate.setDate(firstMonday.getDate() + w * 7 + d);

        const dateStr = cellDate.toISOString().split("T")[0]!;
        const count = dateMap[dateStr] ?? 0;

        let bgColor = "whiteAlpha.100";

        // Only color cells from the selected year
        if (cellDate.getFullYear() === selectedYear && count > 0) {
          if (count < 3) bgColor = "green.800";
          else if (count < 6) bgColor = "green.600";
          else if (count < 10) bgColor = "green.500";
          else if (count < 15) bgColor = "green.400";
          else bgColor = "green.300";
        }

        // The grid is organized with days as rows and weeks as columns
        calendarGrid[d]![w] = {
          date: dateStr,
          count,
          bgColor,
        };
      }
    }

    const totalCount = Object.values(dateMap).reduce(
      (sum, count) => sum + count,
      0
    );

    return {
      calendarGrid,
      dayNames,
      monthLabels,
      totalCount,
      isCurrentYear,
    };
  }, [selectedYear, data, currentYear]);

  const timeDisplayText = calendarData.isCurrentYear
    ? "in the last year"
    : `in ${selectedYear}`;

  const availableYears = useMemo(() => {
    const years = [];
    for (let year = currentYear; year >= joinedYear; year--) {
      years.push(year);
    }
    return years;
  }, [currentYear, joinedYear]);

  return (
    <Box>
      <HStack mb={4} justifyContent="space-between">
        <HStack spacing={3}>
          <Icon as={FaFire} color="brand.primary" w={5} h={5} />
          <Text fontSize="sm" color="whiteAlpha.800">
            <Text as="span" fontWeight="bold" fontSize="md" color="white">
              {calendarData.totalCount}
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
            bg="gray.800"
            color="white"
            borderRadius="lg"
            borderColor="gray.600"
            borderWidth="1px"
            fontWeight="medium"
            _hover={{ borderColor: "blue.300", bg: "gray.650" }}
            _active={{ bg: "gray.650" }}
            rightIcon={<FaChevronDown color="#a3cfff" size={10} />}
          >
            {selectedYear}
          </MenuButton>
          <MenuList
            bg="gray.800"
            borderColor="gray.800"
            borderRadius="lg"
            boxShadow="lg"
            py={1}
            minW="100px"
          >
            {availableYears.map((year) => (
              <MenuItem
                key={year}
                value={year}
                onClick={() => setSelectedYear(year)}
                bg={selectedYear === year ? "blue.900" : "gray.800"}
                borderRadius="lg"
                color="white"
                _hover={{ bg: "gray.600" }}
                fontSize="sm"
              >
                <Text>{year}</Text>
                {selectedYear === year && (
                  <Icon as={FaCheck} ml="auto" boxSize={3} />
                )}
              </MenuItem>
            ))}
          </MenuList>
        </Menu>
      </HStack>

      <Flex align="flex-start" width="100%" ml={5}>
        {/* Left side with day labels */}
        <Box pt={3} width="40px">
          {calendarData.dayNames.map((day, index) => (
            <Box key={index} height="10px" mb={1.5}>
              {(index === 0 || index === 6) && (
                <Text fontSize="xs" color="whiteAlpha.600">
                  {day}
                </Text>
              )}
            </Box>
          ))}
        </Box>

        {/* Calendar container */}
        <Box flex="1" position="relative">
          {/* Month labels */}
          <HStack justifyContent="space-between" width="85%" ml={4} pl={0.5}>
            {calendarData.monthLabels.map((item) => (
              <Text key={item.month} fontSize="xs" color="whiteAlpha.700">
                {item.month}
              </Text>
            ))}
          </HStack>

          {/* Grid of contribution cells */}
          <Box>
            {calendarData.calendarGrid.map((row, rowIndex) => (
              <Flex key={rowIndex} mb={1.5} height="10px">
                {row.map((cell, colIndex) => (
                  <Tooltip
                    key={`${rowIndex}-${colIndex}`}
                    label={
                      cell.date && cell.count > 0
                        ? `${cell.count} submission${
                            cell.count === 1 ? "" : "s"
                          } on ${new Date(cell.date).toLocaleDateString(
                            "en-US",
                            {
                              month: "long",
                              day: "numeric",
                            }
                          )}${getOrdinalSuffix(new Date(cell.date).getDate())}.`
                        : cell.date
                          ? `No submissions on ${new Date(
                              cell.date
                            ).toLocaleDateString("en-US", {
                              month: "long",
                              day: "numeric",
                            })}${getOrdinalSuffix(new Date(cell.date).getDate())}.`
                          : "No submissions"
                    }
                    placement="top"
                    hasArrow
                    bg="blue.200"
                    color="gray.800"
                    fontSize="xs"
                    px={2}
                    py={2}
                  >
                    <Box
                      w="10px"
                      h="10px"
                      bg={cell.bgColor}
                      borderRadius="sm"
                      mr={1}
                    />
                  </Tooltip>
                ))}
              </Flex>
            ))}

            {/* Less/More spectrum */}
            <Flex justify="flex-end" mt={4} width="100%">
              <Flex
                alignItems="center"
                bg="gray.800"
                py={1}
                px={3}
                borderRadius="md"
                mr={3}
              >
                <Text fontSize="xs" color="whiteAlpha.700" mr={2}>
                  Less
                </Text>
                <HStack spacing={1.5}>
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.700" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.600" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.400" />
                  <Box w="10px" h="10px" borderRadius="sm" bg="green.200" />
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
