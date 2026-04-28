import { Badge, Tooltip } from "@chakra-ui/react";
import type { SubmissionModerationStatus } from "@prisma/client";
import {
  SUBMISSION_MODERATION_LABELS,
  SUBMISSION_MODERATION_TOOLTIPS,
} from "~/constants/submissionModeration";

type ModerationStatusBadgeProps = {
  status: SubmissionModerationStatus | null | undefined;
};

export function ModerationStatusBadge({
  status,
}: ModerationStatusBadgeProps): JSX.Element | null {
  if (!status) return null;

  return (
    <Tooltip label={SUBMISSION_MODERATION_TOOLTIPS[status]} hasArrow>
      <Badge
        colorScheme={status === "INVALIDATED" ? "red" : "orange"}
        borderRadius="full"
        px={2}
        py={0.5}
      >
        {SUBMISSION_MODERATION_LABELS[status]}
      </Badge>
    </Tooltip>
  );
}
