export const formatContributionDate = (date: Date): string => {
  const day = date.getDate();
  const month = date.toLocaleString("en-US", { month: "short" });
  const year = date.getFullYear();

  const getDayWithSuffix = (day: number) => {
    if (day > 3 && day < 21) return `${day}th`;
    switch (day % 10) {
      case 1:
        return `${day}st`;
      case 2:
        return `${day}nd`;
      case 3:
        return `${day}rd`;
      default:
        return `${day}th`;
    }
  };

  return `${getDayWithSuffix(day)} ${month} ${year}`;
};
